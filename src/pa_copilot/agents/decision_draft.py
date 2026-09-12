"""Decision-draft worker (design.md §3.3, §3.5, §6.3): synthesizes benefit +
necessity into a PADecision, self-critiques every cited quote against
retrieved_criteria (a hallucinated citation sets needs_replan rather than being
accepted — PR6's reflection.py owns the actual reroute-and-retry enforcement),
and records a critical-importance memory on denial (design.md §6.3:
"denials/appeals = critical") via a direct PolicyStore.put — LangMem's generic
manage_memory tool hardcodes routine and cannot express this."""

from __future__ import annotations

from typing import Awaitable, Callable

from langchain_core.messages import HumanMessage

from pa_copilot.agents._react import (
    WorkerOutputError,
    WorkerRecursionError,
    get_agent_model,
    run_worker_react,
)
from pa_copilot.context.assembly import select_for
from pa_copilot.memory.store import PolicyStore
from pa_copilot.schemas import PADecision
from pa_copilot.state import PACaseState

_SYSTEM_PROMPT = (
    "You are the decision-draft worker for a prior-authorization copilot. Given the "
    "benefit check and medical-necessity assessment, draft a PADecision. Every quote "
    "in cited_criteria MUST be copied verbatim from the retrieved criteria you were "
    "given — never invent or paraphrase a quote. human_review_required should almost "
    "always be True; this is decision support, not a final determination."
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
        prompt = HumanMessage(
            content=(
                f"service_code={request.service_code} covered={benefit.covered} "
                f"criteria_status={necessity.criteria_status} "
                f"unmet_requirements={necessity.unmet_requirements} "
                f"retrieved_criteria={[c.model_dump() for c in state.get('retrieved_criteria', [])]}"
            )
        )
        try:
            _messages, decision = await run_worker_react(
                model or get_agent_model(),
                tools=[],
                system_prompt=_SYSTEM_PROMPT,
                messages=[prompt],
                response_format=PADecision,
            )
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
