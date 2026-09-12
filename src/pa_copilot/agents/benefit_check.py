"""Benefit-check worker (design.md §3.3): calls MCP benefit_lookup, emits a
validated BenefitResult. No untrusted text, no memory read — the request is
already-validated PARequest fields."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Awaitable, Callable

from langchain_core.messages import HumanMessage

from pa_copilot.agents._react import WorkerToolError, get_agent_model, run_worker_react
from pa_copilot.context.assembly import select_for
from pa_copilot.schemas import BenefitResult, ToolFailure
from pa_copilot.state import PACaseState

_BENEFIT_SYSTEM_PROMPT = (
    "You are the benefit-check worker for a prior-authorization copilot. Given a "
    "validated PARequest, call benefit_lookup to determine coverage and whether "
    "prior authorization is required, then emit a BenefitResult."
)


def build_benefit_check_node(
    *, mcp_tools: list | None = None, model=None
) -> Callable[[PACaseState], Awaitable[dict]]:
    tools = list(mcp_tools or [])

    async def _node(state: PACaseState) -> dict:
        view = select_for("benefit_check", state)
        request = view["request"]
        prompt = HumanMessage(
            content=(
                f"member_id={request.member_id} service_code={request.service_code} "
                f"diagnosis_codes={request.diagnosis_codes}"
            )
        )
        try:
            _messages, benefit = await run_worker_react(
                model or get_agent_model(),
                tools,
                system_prompt=_BENEFIT_SYSTEM_PROMPT,
                messages=[prompt],
                response_format=BenefitResult,
            )
        except WorkerToolError as exc:
            failure = ToolFailure(
                tool=exc.tool, error=exc.error, attempt=exc.attempt,
                ts=datetime.now(timezone.utc).isoformat(),
            )
            return {"tool_failures": [failure], "needs_replan": True}

        return {"benefit": benefit}

    return _node
