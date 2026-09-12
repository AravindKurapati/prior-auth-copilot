"""human_review is a thin interrupt() stub — genuinely testing that it PAUSES a
graph run needs a compiled graph + real checkpointer (see test_ac05_checkpointer.py,
Task 6). This test only proves the node calls interrupt() with a sane payload; it
must be exercised from inside a langgraph node execution context (interrupt() reads
an ambient contextvar), so drive it through a minimal throwaway StateGraph rather
than calling the node function bare."""

import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from pa_copilot.agents.human_review import build_human_review_node
from pa_copilot.state import PACaseState


@pytest.mark.asyncio
async def test_human_review_interrupts_then_resumes():
    node = build_human_review_node()
    g = StateGraph(dict)
    g.add_node("human_review", node)
    g.add_edge(START, "human_review")
    g.add_edge("human_review", END)
    compiled = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-test"}}

    first = await compiled.ainvoke({"case_id": "case-0001"}, config=cfg)
    assert "__interrupt__" in first

    second = await compiled.ainvoke(Command(resume="approved-by-reviewer"), config=cfg)
    assert "__interrupt__" not in second


@pytest.mark.asyncio
async def test_resume_resets_hop_and_replan_counters():
    # StateGraph(PACaseState), not StateGraph(dict) like the test above: verified
    # empirically this session that a bare `dict` graph schema combined with this
    # node's `state: PACaseState`-annotated parameter produces incorrect merge
    # behavior on resume (LangGraph appears to introspect the node's own type
    # annotation and apply schema-specific handling that conflicts with a
    # generic-dict graph schema -- a real reset update silently failed to apply
    # under StateGraph(dict) in a minimal repro, but applied correctly under
    # StateGraph(PACaseState)). Using the real schema also matches what
    # graph.py::make_graph actually does in production.
    node = build_human_review_node()
    g = StateGraph(PACaseState)
    g.add_node("human_review", node)
    g.add_edge(START, "human_review")
    g.add_edge("human_review", END)
    compiled = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-resume-reset"}}

    await compiled.ainvoke(
        {"case_id": "case-0001", "supervisor_hops": 9, "replan_count": 2}, config=cfg
    )
    second = await compiled.ainvoke(Command(resume="approved-by-reviewer"), config=cfg)
    assert second["supervisor_hops"] == 0
    assert second["replan_count"] == 0
