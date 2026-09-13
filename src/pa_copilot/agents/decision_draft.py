"""Decision-draft worker (design.md §3.3, §3.5, §6.3): synthesizes benefit +
necessity into a PADecision, self-critiques every cited quote against
retrieved_criteria (a hallucinated citation sets needs_replan rather than being
accepted — PR6's reflection.py owns the actual reroute-and-retry enforcement),
and records a critical-importance memory on denial (design.md §6.3:
"denials/appeals = critical") via a direct PolicyStore.put — LangMem's generic
manage_memory tool hardcodes routine and cannot express this. PR8 good-to-have:
also handles necessity=None as a valid input when the benefit check alone is
determinative (design.md §3.3's fast-path, `_benefit_is_determinative`) —
supervisor.hard_route routes straight here in that case, skipping
medical_necessity entirely."""

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
from pa_copilot.context.assembly import select_for
from pa_copilot.memory.store import PolicyStore
from pa_copilot.reflection import attribute_tool_errors, run_worker_react_resilient
from pa_copilot.schemas import PADecision
from pa_copilot.state import PACaseState

_SYSTEM_PROMPT = (
    "You are the decision-draft worker for a prior-authorization copilot. Given the "
    "benefit check and medical-necessity assessment, draft a PADecision. Every quote "
    "in cited_criteria MUST be copied verbatim from the retrieved criteria you were "
    "given — never invent or paraphrase a quote. human_review_required should almost "
    "always be True; this is decision support, not a final determination. If no "
    "medical-necessity assessment is provided, the benefit check alone was "
    "determinative (service not covered, or prior authorization not required) — "
    "draft the decision directly from the benefit result and leave cited_criteria "
    "empty."
)


def _benefit_is_determinative(benefit) -> bool:
    """design.md §3.3's fast-path: a benefit that is not covered, or does not
    require prior auth at all, needs no medical-necessity assessment to decide."""
    return benefit is not None and not (
        getattr(benefit, "covered", True) and getattr(benefit, "requires_pa", True)
    )


def _citations_supported(decision: PADecision, retrieved_criteria: list) -> bool:
    known_quotes = {c.quote for c in retrieved_criteria}
    return all(c.quote in known_quotes for c in decision.cited_criteria)


def build_decision_draft_node(
    *, store: PolicyStore, model=None
) -> Callable[[PACaseState], Awaitable[dict]]:
    async def _node(state: PACaseState) -> dict:
        view = select_for("decision_draft", state)
        request, benefit, necessity = view["request"], view["benefit"], view["necessity"]
        fast_path = necessity is None and _benefit_is_determinative(benefit)
        if necessity is None and not fast_path:
            # Belt-and-suspenders (PR5b final-review Fix B, part 2): the
            # supervisor's hard_route guardrail is what actually prevents this
            # in practice (necessity-None routes to medical_necessity before
            # this node can ever run, UNLESS the benefit fast-path below
            # applies), but this node must not itself crash dereferencing a
            # None necessity if it is ever reached out of order.
            return {"needs_replan": True}
        if fast_path:
            prompt = HumanMessage(
                content=(
                    f"service_code={request.service_code} covered={benefit.covered} "
                    f"requires_pa={benefit.requires_pa} "
                    "medical necessity was not assessed: the benefit check alone is "
                    "determinative (not covered, or no prior authorization required), "
                    "so draft the decision directly from the benefit result."
                )
            )
        else:
            prompt = HumanMessage(
                content=(
                    f"service_code={request.service_code} covered={benefit.covered} "
                    f"criteria_status={necessity.criteria_status} "
                    f"unmet_requirements={necessity.unmet_requirements} "
                    f"retrieved_criteria={[c.model_dump() for c in state.get('retrieved_criteria', [])]}"
                )
            )
        try:
            _messages, decision = await run_worker_react_resilient(
                model or get_agent_model(),
                attribute_tool_errors([]),
                system_prompt=_SYSTEM_PROMPT,
                messages=[prompt],
                response_format=PADecision,
                # Deferred like `model or get_agent_model()` above -- see
                # intake.py's identical comment for why.
                lite_model=get_lite_agent_model() if model is None else None,
            )
        except WorkerToolError as exc:
            # Final whole-branch review finding (PR6): run_worker_react_resilient's
            # worker_timeout_seconds wraps the WHOLE turn, model call included --
            # decision_draft has zero tools, but a slow/hung model response still
            # raises WorkerToolError(tool="_worker_turn_", ...) on timeout. Before
            # this fix it was the only one of the four workers not catching
            # WorkerToolError, so a timeout here escaped graph.ainvoke() uncaught,
            # directly contradicting NFR-07's "never an unhandled exception" claim.
            return tool_failure_update(exc)
        except (WorkerOutputError, WorkerRecursionError):
            return {"needs_replan": True}

        if not _citations_supported(decision, state.get("retrieved_criteria", [])):
            return {"needs_replan": True}

        if decision.disposition == "deny":
            store.put(
                ("pa", "member", state["member_id"]),
                f"decision:{state['case_id']}",
                {
                    "content": (
                        f"Prior-auth denied for service {request.service_code}: "
                        f"{decision.reviewer_summary}"
                    ),
                    "importance": "critical",
                },
            )

        return {"decision": decision}

    return _node
