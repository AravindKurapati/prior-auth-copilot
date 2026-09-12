"""human_review (design.md §3.3): terminal stub representing handoff to a human
reviewer. interrupt() pauses graph execution; the checkpointer persists the state
so the case can be resumed later, in another process (AC-05)."""

from __future__ import annotations

from typing import Awaitable, Callable

from langgraph.types import interrupt

from pa_copilot.context.assembly import write_working_memory
from pa_copilot.state import PACaseState


def build_human_review_node() -> Callable[[PACaseState], Awaitable[dict]]:
    async def _node(state: PACaseState) -> dict:
        payload = {
            "case_id": state.get("case_id"),
            "request": state["request"].model_dump() if state.get("request") else None,
            "necessity": state["necessity"].model_dump() if state.get("necessity") else None,
            "decision": state["decision"].model_dump() if state.get("decision") else None,
            "reason": "human review required",
        }
        resume_value = interrupt(payload)
        return write_working_memory(state, "human_review_resume_value", resume_value)

    return _node
