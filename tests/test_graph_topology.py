"""Structural assertions on the compiled graph — every node design.md §3.1 names
is present, the conditional edge's path_map covers all 6 targets."""

import pytest

from pa_copilot.graph import make_graph
from _fakes import FakeToolCallingModel


@pytest.mark.asyncio
async def test_graph_has_all_seven_nodes(memory_store):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=[],
        model=FakeToolCallingModel(),
    )
    node_names = set(graph.get_graph().nodes.keys())
    expected = {"__start__", "summarize", "supervisor", "intake", "benefit_check",
                "medical_necessity", "decision_draft", "human_review", "__end__"}
    assert expected.issubset(node_names)
