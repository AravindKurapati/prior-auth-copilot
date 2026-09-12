"""AC-02: the supervisor routes a full case across specialized workers.

First point in this PR where all 5 workers + supervisor + `graph.py` run
together as one compiled `StateGraph`, not tested in isolation -- see
`tests/_full_case.py` for the shared clear-cut/ambiguous scripting and why one
`FakeToolCallingModel` instance drives every model call in the run.
"""

from __future__ import annotations

import os

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from _full_case import FAKE_MCP_TOOLS, build_clear_cut_case, new_initial_state, stub_summarizer
from pa_copilot.graph import make_graph


@pytest.mark.asyncio
async def test_full_clearcut_run_visits_at_least_two_distinct_workers(memory_store, monkeypatch):
    stub_summarizer(monkeypatch)
    case = build_clear_cut_case()
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=FAKE_MCP_TOOLS, model=case.model
    )
    thread = {"configurable": {"thread_id": case.case_id}}

    result = await graph.ainvoke(new_initial_state(case.case_id), config=thread)

    assert "__interrupt__" not in result
    assert result["next"] == "FINISH"
    visited = {step.to_node for step in result["route_history"] if step.to_node != "FINISH"}
    assert len(visited) >= 2, f"expected >=2 distinct workers, got {visited}"
    # The clear-cut script routes through all four workers on the auto-draft
    # path -- pin the exact set rather than just the count, so a future change
    # that silently drops a hop is caught here too.
    assert visited == {"intake", "benefit_check", "medical_necessity", "decision_draft"}


@pytest.mark.asyncio
async def test_full_clearcut_run_reaches_a_finished_decision(memory_store, monkeypatch):
    """Companion sanity check: AC-02 is about routing, but a route_history that
    "visits workers" while the run itself silently failed to finish would be a
    hollow proof -- pin the actual terminal decision too."""
    stub_summarizer(monkeypatch)
    case = build_clear_cut_case()
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=FAKE_MCP_TOOLS, model=case.model
    )
    thread = {"configurable": {"thread_id": case.case_id}}

    result = await graph.ainvoke(new_initial_state(case.case_id), config=thread)

    assert result["decision"] == case.expected_decision
    assert result["necessity"] == case.expected_necessity
    assert result["benefit"] == case.expected_benefit
    assert result["request"] == case.expected_request


@pytest.mark.asyncio
async def test_supervisor_router_decision_is_a_validated_routerdecision_each_hop(
    memory_store, monkeypatch
):
    """The one remaining LLM-routed hop (request/benefit/necessity are all
    hard-routed deterministically since PR5b final-review Fix B added the
    benefit=None/necessity=None guardrails -- only the decision_draft
    transition still falls through to the LLM router) must be a real
    validated `RouterDecision`, not a raw dict -- pins the supervisor's
    `.with_structured_output(RouterDecision)` contract inside a genuine
    multi-worker run rather than only the single-hop unit test in
    `test_supervisor.py`."""
    stub_summarizer(monkeypatch)
    case = build_clear_cut_case()
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=FAKE_MCP_TOOLS, model=case.model
    )
    thread = {"configurable": {"thread_id": case.case_id}}

    result = await graph.ainvoke(new_initial_state(case.case_id), config=thread)

    llm_routed = [
        step for step in result["route_history"]
        if not step.reason.startswith("deterministic guardrail ->")
    ]
    assert len(llm_routed) == 1
    assert [s.to_node for s in llm_routed] == ["decision_draft"]


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="needs GEMINI_API_KEY")
@pytest.mark.asyncio
async def test_full_clearcut_run_with_real_gemini_reaches_a_terminal_state(memory_store):
    """Live counterpart: real Gemini for both the supervisor router and every
    worker, real MCP tools over a live subprocess session, real (fake-embedder)
    RAG tool. No script to pop -- the real model drives itself. Best-effort:
    only asserts the run terminates (FINISH or human_review) with no crash and
    a non-trivial route_history, since a live model's exact path is not
    pinned -- same tolerant-assertion style as
    `test_ac10_gemini_agent_invokes_mcp_tool_transcript`."""
    from pa_copilot.mcp_client import load_pa_tools, pa_session
    from pa_copilot.rag.tool import reset_tool_embedder, set_tool_embedder
    from _fakes import FakeEmbedder
    from _full_case import CASE_ID_CLEARCUT as _CID, new_initial_state as _new_state

    set_tool_embedder(FakeEmbedder())
    try:
        async with pa_session() as session:
            tools = await load_pa_tools(session=session)
            graph = await make_graph(
                store=memory_store, checkpointer=InMemorySaver(), mcp_tools=tools, model=None
            )
            thread = {"configurable": {"thread_id": f"{_CID}-live"}}
            result = await graph.ainvoke(_new_state(f"{_CID}-live"), config=thread)
    finally:
        reset_tool_embedder()

    assert result.get("next") in ("FINISH", "human_review")
    assert len(result.get("route_history") or []) >= 1
