"""Medical-necessity worker (design.md §3.3, §4.1, §7): calls MCP criteria_check,
decides — agentically, via the system prompt teaching design.md §4.1's status
mapping, not a hard code gate — whether to call the agentic RAG tool, emits a
validated NecessityAssessment."""

from __future__ import annotations

from typing import Awaitable, Callable

from langchain_core.messages import HumanMessage

from pa_copilot.agents._react import (
    WorkerOutputError,
    WorkerRecursionError,
    WorkerToolError,
    get_agent_model,
    run_worker_react,
    tool_failure_update,
)
from pa_copilot.context.assembly import select_for
from pa_copilot.rag.tool import search_clinical_guidance
from pa_copilot.schemas import NecessityAssessment
from pa_copilot.state import PACaseState

_SYSTEM_PROMPT = (
    "You are the medical-necessity worker for a prior-authorization copilot. Call "
    "criteria_check to get the mechanical policy screen. Its `status` field means: "
    "not_found -> no policy exists, you cannot mechanically assess, treat as "
    "indeterminate with policy_id=None; excluded -> a diagnosis is a documented "
    "exclusion, a strong signal but you should still confirm against the clinical "
    "summary before finalizing not_met; indeterminate -> a policy exists but only "
    "narrative clinical guidance can verify the conditions. Call "
    "search_clinical_guidance whenever criteria_check's status is not_found or "
    "indeterminate, or when there are unmet requirements that need narrative "
    "interpretation, or when you need to confirm an exclusion. Do not call it for a "
    "clearly met case with no unmet requirements. Emit a NecessityAssessment."
)


def build_medical_necessity_node(
    *, mcp_tools: list | None = None, model=None
) -> Callable[[PACaseState], Awaitable[dict]]:
    tools = [*(mcp_tools or []), search_clinical_guidance]

    async def _node(state: PACaseState) -> dict:
        view = select_for("medical_necessity", state)
        request, benefit = view["request"], view["benefit"]
        prompt = HumanMessage(
            content=(
                f"service_code={request.service_code} diagnosis_codes={request.diagnosis_codes} "
                f"clinical_summary={request.clinical_summary!r} covered={benefit.covered} "
                f"requires_pa={benefit.requires_pa}"
            )
        )
        try:
            _messages, necessity = await run_worker_react(
                model or get_agent_model(),
                tools,
                system_prompt=_SYSTEM_PROMPT,
                messages=[prompt],
                response_format=NecessityAssessment,
            )
        except WorkerToolError as exc:
            return tool_failure_update(exc)
        except (WorkerOutputError, WorkerRecursionError):
            return {"needs_replan": True}

        return {"necessity": necessity, "retrieved_criteria": necessity.citations}

    return _node
