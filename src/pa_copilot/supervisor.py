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

Two more deterministic guardrails sit in the same tier, immediately after
request-None (PR5b final-review fix wave, Fix B): benefit-None -> benefit_check
and necessity-None -> medical_necessity. The LLM router (RouterDecision.next)
is a free choice over all five workers with no ordering guarantee, so nothing
previously stopped it picking medical_necessity before benefit_check had set
state["benefit"], or decision_draft before medical_necessity had set
state["necessity"] -- both reproduced as uncaught AttributeErrors in the two
downstream workers. These two rules force the natural prerequisite order
deterministically, without disturbing any guardrail already ordered above
them.

PR8 good-to-have: design.md §3.3's documented decision_draft short-circuit
(skipping medical_necessity when benefit.covered=False or
benefit.requires_pa=False) is now re-enabled -- the necessity-None rule
checks benefit first and routes straight to decision_draft when the benefit
check alone is determinative; decision_draft.py's `_benefit_is_determinative`
handles necessity=None as a valid input in exactly that case.
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
    if (state.get("tool_failures") or state.get("needs_replan")) and state.get(
        "replan_count", 0
    ) >= s.max_replans:
        return "human_review"
    if state.get("decision") is not None and not state.get("needs_replan"):
        return "FINISH"
    if state.get("request") is None:
        return "intake"
    if state.get("benefit") is None:
        return "benefit_check"
    if state.get("necessity") is None:
        benefit = state.get("benefit")
        if not (getattr(benefit, "covered", True) and getattr(benefit, "requires_pa", True)):
            # PR8 good-to-have: design.md §3.3's fast-path, now re-enabled --
            # decision_draft.py handles necessity=None when the benefit check
            # alone is determinative (not covered, or no PA required).
            return "decision_draft"
        return "medical_necessity"
    return None


def _missing_checklist(state: PACaseState) -> list[str]:
    return [f for f in ("request", "benefit", "necessity", "decision") if state.get(f) is None]


async def route_with_llm(
    state: PACaseState, *, model=None, settings: Settings | None = None
) -> RouterDecision:
    model = model or get_agent_model()
    s = settings or get_settings()
    summary = (state.get("context") or {}).get("running_summary")
    summary_text = getattr(summary, "summary", None) or "no summary yet"
    history = state.get("route_history") or []
    hint = ""
    if state.get("needs_replan"):
        necessity = state.get("necessity")
        if necessity is not None and (
            necessity.confidence < s.tau or necessity.criteria_status == "indeterminate"
        ):
            hint = (
                f"\nReflection hint: the last medical_necessity assessment was "
                f"low-confidence (confidence={necessity.confidence:.2f}, "
                f"status={necessity.criteria_status}). Consider routing back to "
                "medical_necessity to reassess with broader retrieval, or escalate to "
                "human_review if this has already been retried."
            )
        elif state.get("tool_failures"):
            hint = (
                "\nReflection hint: the last worker call failed a tool call. Routing "
                "back to the same phase will retry it; escalate to human_review if "
                "this has already been retried multiple times."
            )
    prompt = (
        "You are the routing supervisor for a prior-authorization case.\n"
        f"Case summary so far: {summary_text}\n"
        f"Still missing: {_missing_checklist(state)}\n"
        f"Recent routing history: {[h for h in history[-5:]]}"
        f"{hint}\n"
        "Choose the next node."
    )
    return await model.with_structured_output(RouterDecision).ainvoke([("user", prompt)])


def build_supervisor_node(*, model=None, settings: Settings | None = None) -> Callable:
    s = settings or get_settings()

    async def _node(state: PACaseState) -> dict:
        was_replanning = bool(state.get("needs_replan"))
        replan_count = state.get("replan_count", 0) + (1 if was_replanning else 0)
        effective_state = {**state, "replan_count": replan_count} if was_replanning else state

        target = hard_route(effective_state, settings=s)
        if target is not None:
            reason = f"deterministic guardrail -> {target}"
        else:
            decision = await route_with_llm(effective_state, model=model, settings=s)
            target = decision.next
            reason = decision.rationale

        step = RouteStep(
            from_node="supervisor", to_node=target, reason=reason,
            ts=datetime.now(timezone.utc).isoformat(),
        )
        update = {
            "next": target,
            "route_history": [step],
            "supervisor_hops": state.get("supervisor_hops", 0) + 1,
        }
        if was_replanning:
            update["replan_count"] = replan_count
            update["needs_replan"] = False
        return update

    return _node
