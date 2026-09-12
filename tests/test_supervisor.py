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
    assert hard_route({"request": {}}) is None


@pytest.mark.asyncio
async def test_supervisor_node_uses_llm_router_when_no_hard_rule_matches():
    fake = FakeToolCallingModel(structured_responses=[
        RouterDecision(next="benefit_check", rationale="request captured, need benefit check")
    ])
    node = build_supervisor_node(model=fake)
    update = await node({"request": {"service_code": "72148"}, "route_history": [], "supervisor_hops": 0})
    assert update["next"] == "benefit_check"
    assert len(update["route_history"]) == 1
    assert update["supervisor_hops"] == 1


@pytest.mark.asyncio
async def test_supervisor_node_uses_hard_rule_without_calling_model():
    node = build_supervisor_node(model=None)  # would blow up if it tried a real LLM call
    update = await node({})
    assert update["next"] == "intake"
