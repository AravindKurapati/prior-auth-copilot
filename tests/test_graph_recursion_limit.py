"""Fix A (PR5b final-review): settings.recursion_limit must be bound onto the
compiled graph via .with_config(...), not left at LangGraph's default of 25 --
otherwise a long-running case can hit GraphRecursionError around
supervisor_hops == 8, well before config/routing.yaml's own max_hops=12
guardrail ever gets a chance to fire.

Verified empirically (langgraph 1.0.1) that CompiledStateGraph.with_config(...)
returns a new CompiledStateGraph whose bound config is inspectable via
`.config` (a plain dict, e.g. {"recursion_limit": N, "configurable": {}}), and
that a run exceeding a small bound recursion_limit raises GraphRecursionError
-- both are exercised below."""

import dataclasses

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from _fakes import FakeToolCallingModel
from pa_copilot.config import get_settings
from pa_copilot.graph import make_graph


@pytest.mark.asyncio
async def test_make_graph_binds_settings_recursion_limit(memory_store):
    """The compiled graph's bound config carries settings.recursion_limit (40
    from config/routing.yaml) -- not LangGraph's default of 25."""
    settings = get_settings()
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=[],
        model=FakeToolCallingModel(),
    )
    assert graph.config["recursion_limit"] == settings.recursion_limit
    assert settings.recursion_limit == 40  # config/routing.yaml's configured value


@pytest.mark.asyncio
async def test_make_graph_honors_a_custom_recursion_limit_override(memory_store):
    """A caller-supplied settings object's recursion_limit is what actually
    gets bound (not a hardcoded constant) -- proven with a deliberately tiny
    override rather than assuming the default just happens to match."""
    settings = dataclasses.replace(get_settings(), recursion_limit=2)
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=[],
        model=FakeToolCallingModel(), settings=settings,
    )
    assert graph.config["recursion_limit"] == 2


def test_with_config_actually_enforces_a_bound_recursion_limit():
    """Direct mechanism proof, isolated from the full pa_copilot graph (which
    needs real request/benefit/necessity state to run past a couple of
    supervisor hops before it would ever recurse deeply): a trivial
    self-looping graph bound via the same .with_config({"recursion_limit":
    ...}) mechanism make_graph uses raises GraphRecursionError well before
    LangGraph's default limit of 25 would -- the cheapest, fastest way to
    prove the binding mechanism itself actually constrains execution."""
    g = StateGraph(dict)
    g.add_node("loop", lambda s: {"n": s.get("n", 0) + 1})
    g.add_conditional_edges(
        "loop", lambda s: END if s["n"] >= 100 else "loop", {"loop": "loop", END: END}
    )
    g.add_edge(START, "loop")
    compiled = g.compile().with_config({"recursion_limit": 3})

    with pytest.raises(GraphRecursionError):
        compiled.invoke({"n": 0})
