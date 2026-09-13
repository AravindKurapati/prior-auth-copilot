"""AC-12: reflection/self-healing loop, full in-graph proof (design.md §3.5).
Drives all four scenarios (tool-failure recover/exhaust, low-confidence
recover/exhaust) through the REAL compiled graph (pa_copilot.graph.make_graph),
scripted via tests/_reflection_case.py -- deterministic, no API key needed."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from _full_case import stub_summarizer
from _reflection_case import (
    build_low_confidence_exhausts_case,
    build_low_confidence_recovers_case,
    build_tool_failure_exhausts_case,
    build_tool_failure_recovers_case,
    seed_request_and_benefit,
    seed_request_only,
)
from pa_copilot.config import get_settings
from pa_copilot.graph import make_graph


@pytest.mark.asyncio
async def test_tool_failure_recovers_and_reaches_finish(memory_store, monkeypatch):
    stub_summarizer(monkeypatch)
    case = build_tool_failure_recovers_case()
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=case.mcp_tools, model=case.model
    )
    thread = {"configurable": {"thread_id": case.case_id}}
    result = await graph.ainvoke(seed_request_only(case.case_id), config=thread)

    assert "__interrupt__" not in result
    assert result["next"] == "FINISH"
    assert result["decision"].disposition == "approve"
    # benefit_check visited at least twice: the failing attempt + the
    # recovering reroute -- proves the OUTER supervisor-level reroute, not
    # just tenacity's inner retry.
    benefit_check_visits = [s for s in result["route_history"] if s.to_node == "benefit_check"]
    assert len(benefit_check_visits) >= 2
    assert len(result["tool_failures"]) == 1
    assert result["tool_failures"][0].tool == "benefit_lookup"
    assert result["replan_count"] == 1


@pytest.mark.asyncio
async def test_tool_failure_exhausts_and_reaches_human_review(memory_store, monkeypatch):
    stub_summarizer(monkeypatch)
    settings = get_settings()
    case = build_tool_failure_exhausts_case()
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=case.mcp_tools, model=case.model
    )
    thread = {"configurable": {"thread_id": case.case_id}}
    # The whole ainvoke call must complete cleanly -- NFR-07's "never an
    # unhandled exception" requirement -- the only way a persistently failing
    # tool can end a run is a clean route to human_review.
    result = await graph.ainvoke(seed_request_only(case.case_id), config=thread)

    assert "__interrupt__" in result
    assert result["replan_count"] == settings.max_replans
    assert len(result["tool_failures"]) == settings.max_replans
    assert all(f.tool == "benefit_lookup" for f in result["tool_failures"])


@pytest.mark.asyncio
async def test_low_confidence_recovers_and_reaches_finish(memory_store, monkeypatch):
    stub_summarizer(monkeypatch)
    case = build_low_confidence_recovers_case()
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=case.mcp_tools, model=case.model
    )
    thread = {"configurable": {"thread_id": case.case_id}}
    result = await graph.ainvoke(seed_request_and_benefit(case.case_id), config=thread)

    assert "__interrupt__" not in result
    assert result["next"] == "FINISH"
    assert result["decision"].disposition == "approve"
    necessity_visits = [s for s in result["route_history"] if s.to_node == "medical_necessity"]
    assert len(necessity_visits) >= 2
    assert result["necessity"].confidence >= 0.9  # the SECOND (recovered) assessment won
    assert result["replan_count"] == 1
    # The loop-back was a real LLM router CHOICE (design.md's "supervisor loops
    # back with a hint"), not a forced hard_route rule -- proven by the fact
    # this scenario needed a scripted RouterDecision at all (see
    # _reflection_case.py's build_low_confidence_recovers_case).
    reroute_step = [s for s in necessity_visits][-1]
    assert "low confidence" in reroute_step.reason.lower() or "reassess" in reroute_step.reason.lower()


@pytest.mark.asyncio
async def test_low_confidence_exhausts_and_reaches_human_review(memory_store, monkeypatch):
    stub_summarizer(monkeypatch)
    settings = get_settings()
    case = build_low_confidence_exhausts_case()
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=case.mcp_tools, model=case.model
    )
    thread = {"configurable": {"thread_id": case.case_id}}
    result = await graph.ainvoke(seed_request_and_benefit(case.case_id), config=thread)

    assert "__interrupt__" in result
    assert result["replan_count"] == settings.max_replans
    assert result["necessity"].confidence < settings.tau
    # No ToolFailure at all -- this is the needs_replan-alone cap path
    # (Task 3's broadened hard_route check), distinct from the tool-failure
    # scenarios above.
    assert result.get("tool_failures", []) == []
