"""AC-03: conditional edges route on state -- the SAME compiled graph, given a
clear-cut state, takes the auto-draft path to `decision_draft`/`FINISH`; given
an ambiguous (low-confidence, indeterminate necessity) state, it routes to
`human_review` instead. See `tests/_full_case.py` for the shared scripting.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from _full_case import (
    FAKE_MCP_TOOLS,
    build_ambiguous_case,
    build_clear_cut_case,
    fake_rag_embedder,
    new_initial_state,
    stub_summarizer,
)
from pa_copilot.graph import make_graph


@pytest.mark.asyncio
async def test_same_compiled_graph_clearcut_reaches_decision_draft(memory_store, monkeypatch):
    stub_summarizer(monkeypatch)
    case = build_clear_cut_case()
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=FAKE_MCP_TOOLS, model=case.model
    )
    thread = {"configurable": {"thread_id": case.case_id}}

    result = await graph.ainvoke(new_initial_state(case.case_id), config=thread)

    assert "__interrupt__" not in result
    assert result["next"] == "FINISH"
    assert result["decision"] is not None
    assert result["decision"].disposition == "approve"
    assert any(step.to_node == "decision_draft" for step in result["route_history"])
    assert not any(step.to_node == "human_review" for step in result["route_history"])


@pytest.mark.asyncio
async def test_same_compiled_graph_ambiguous_reaches_human_review_not_decision_draft(
    memory_store, monkeypatch
):
    stub_summarizer(monkeypatch)
    case = build_ambiguous_case()
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=FAKE_MCP_TOOLS, model=case.model
    )
    thread = {"configurable": {"thread_id": case.case_id}}

    with fake_rag_embedder():
        result = await graph.ainvoke(new_initial_state(case.case_id), config=thread)

    assert "__interrupt__" in result
    interrupt_payload = result["__interrupt__"][0].value
    assert interrupt_payload["reason"] == "human review required"
    # The case never reached decision_draft: no decision was ever drafted, and
    # decision_draft never appears in the routing trail.
    assert interrupt_payload["decision"] is None
    assert result.get("decision") is None
    assert any(step.to_node == "human_review" for step in result["route_history"])
    assert not any(step.to_node == "decision_draft" for step in result["route_history"])
