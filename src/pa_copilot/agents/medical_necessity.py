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
    get_lite_agent_model,
    tool_failure_update,
)
from pa_copilot.config import get_settings
from pa_copilot.context.assembly import select_for
from pa_copilot.rag.tool import search_clinical_guidance
from pa_copilot.reflection import attribute_tool_errors, run_worker_react_resilient
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
    # search_clinical_guidance is a module-level singleton reused by every
    # build_medical_necessity_node() call; attribute_tool_errors mutates a
    # tool in place (sets .func/.coroutine wrappers + the _pa_attributed
    # marker), so wrapping the singleton directly would leak that mutation
    # process-wide -- including into unrelated tests that take their own
    # `search_clinical_guidance.model_copy(...)` and expect a pristine,
    # unwrapped tool (the marker survives model_copy()). A per-build copy
    # keeps the wrapping local to this node's own tools list.
    tools = attribute_tool_errors([*(mcp_tools or []), search_clinical_guidance.model_copy()])

    async def _node(state: PACaseState) -> dict:
        if state.get("benefit") is None:
            # Belt-and-suspenders (PR5b final-review Fix B, part 2): the
            # supervisor's hard_route guardrail is what actually prevents this
            # in practice (benefit-None routes to benefit_check before this
            # node can ever run), but this node must not itself crash
            # dereferencing a None benefit if it is ever reached out of order.
            return {"needs_replan": True}
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
            _messages, necessity = await run_worker_react_resilient(
                model or get_agent_model(),
                tools,
                system_prompt=_SYSTEM_PROMPT,
                messages=[prompt],
                response_format=NecessityAssessment,
                # Deferred like `model or get_agent_model()` above: only build a
                # real lite fallback when the caller didn't already supply a
                # substitute `model` (tests inject a FakeToolCallingModel here
                # and have no corresponding real lite backing -- constructing
                # get_lite_agent_model() unconditionally would eagerly hit
                # real Google credential resolution on every call, fake-model
                # tests included, since Python evaluates this argument before
                # run_worker_react_resilient ever runs).
                lite_model=get_lite_agent_model() if model is None else None,
            )
        except WorkerToolError as exc:
            return tool_failure_update(exc)
        except (WorkerOutputError, WorkerRecursionError):
            return {"needs_replan": True}

        update = {"necessity": necessity, "retrieved_criteria": necessity.citations}
        # AC-12 low-confidence reflection trigger (design.md §3.5): flag for
        # the supervisor's bookkeeping (replan_count increment + cap), but do
        # NOT withhold `necessity` from state -- the supervisor's LLM router
        # already has a fall-through path for this exact state (necessity set,
        # decision None, no hard_route rule matches) and decides whether to
        # loop back here or escalate, exactly as design.md's "supervisor loops
        # back with a hint" implies. Withholding necessity would be a second,
        # conflicting mechanism and would regress tests/_full_case.py's
        # ambiguous-case fixture, which relies on this exact low-confidence
        # NecessityAssessment still being visible to the (scripted) router.
        settings = get_settings()
        if necessity.confidence < settings.tau or necessity.criteria_status == "indeterminate":
            update["needs_replan"] = True
        return update

    return _node
