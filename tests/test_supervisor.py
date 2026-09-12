"""Supervisor: deterministic guardrails checked before any LLM call, else a
single .with_structured_output(RouterDecision) call. No tools — the supervisor
never calls MCP/RAG directly (design.md §3.2)."""

import pytest

from _fakes import FakeToolCallingModel
from pa_copilot.config import get_settings
from pa_copilot.schemas import RouterDecision
from pa_copilot.supervisor import build_supervisor_node, hard_route


def test_hard_route_no_request_goes_to_intake():
    assert hard_route({}) == "intake"


def test_hard_route_decision_finished_goes_to_finish():
    assert hard_route({"request": {}, "decision": {}, "needs_replan": False}) == "FINISH"


def test_hard_route_exhausted_replans_goes_to_human_review():
    s = get_settings()
    state = {"request": {}, "tool_failures": [{"tool": "x"}], "replan_count": s.max_replans}
    assert hard_route(state, settings=s) == "human_review"


def test_hard_route_hop_cap_wins_over_finished_decision():
    s = get_settings()
    state = {
        "request": {}, "decision": {}, "needs_replan": False,
        "supervisor_hops": s.max_hops,
    }
    assert hard_route(state, settings=s) == "human_review"


def test_hard_route_falls_through_to_none_when_nothing_matches():
    # request/benefit/necessity must all be truthy now (PR5b final-review Fix
    # B added benefit-None -> benefit_check and necessity-None ->
    # medical_necessity guardrails between request-None and the fall-through),
    # otherwise this state would now match one of those new rules instead of
    # falling through -- populating them here still proves the same thing
    # this test always proved: with every prerequisite present, hard_route
    # truly has no deterministic rule left to apply and defers to the LLM
    # router.
    assert hard_route({"request": {}, "benefit": {}, "necessity": {}}) is None


def test_hard_route_missing_benefit_goes_to_benefit_check():
    assert hard_route({"request": {}, "benefit": None}) == "benefit_check"


def test_hard_route_missing_necessity_goes_to_medical_necessity():
    assert hard_route({"request": {}, "benefit": {}, "necessity": None}) == "medical_necessity"


def test_hard_route_finished_decision_wins_over_missing_request():
    """Unreachable in the current topology (decision can only be set downstream of
    intake); pins the accepted guardrail-order ruling rather than leaving it untested."""
    assert hard_route({"decision": {}, "needs_replan": False}) == "FINISH"


@pytest.mark.asyncio
async def test_supervisor_node_uses_llm_router_when_no_hard_rule_matches():
    # request/benefit/necessity must all be truthy (PR5b final-review Fix B):
    # otherwise the new benefit-None/necessity-None guardrails would fire
    # deterministically and this test would stop exercising the LLM router
    # path at all while still passing (the fake's target happens to match
    # what a hard rule would also pick). Route to decision_draft instead so a
    # coincidental match with a hard rule can't mask that.
    fake = FakeToolCallingModel(structured_responses=[
        RouterDecision(next="decision_draft", rationale="benefit + necessity captured, ready to draft")
    ])
    node = build_supervisor_node(model=fake)
    update = await node({
        "request": {"service_code": "72148"}, "benefit": {}, "necessity": {},
        "route_history": [], "supervisor_hops": 0,
    })
    assert update["next"] == "decision_draft"
    assert len(update["route_history"]) == 1
    assert update["supervisor_hops"] == 1


@pytest.mark.asyncio
async def test_supervisor_node_uses_hard_rule_without_calling_model():
    node = build_supervisor_node(model=None)  # would blow up if it tried a real LLM call
    update = await node({})
    assert update["next"] == "intake"
