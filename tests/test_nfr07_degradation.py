"""NFR-07: graceful degradation on tool/model failure -- timeouts, retries,
explicit exit conditions. A persistently failing (or hanging) tool must end at
human_review, never an unhandled exception (design.md §3.6)."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from _fakes import FakeToolCallingModel, ai_tool_call
from _full_case import MEMBER_ID, SERVICE_CODE, stub_summarizer
from _reflection_case import build_tool_failure_exhausts_case, seed_request_only
from pa_copilot.config import get_settings
from pa_copilot.graph import make_graph


@pytest.mark.asyncio
async def test_persistent_tool_failure_ends_at_human_review_not_unhandled_exception(
    memory_store, monkeypatch
):
    """NFR-07's own named evidence for the "exit condition" half -- the exact
    mechanism is Task 5's build_tool_failure_exhausts_case (tests/_reflection_
    case.py), re-run here under this file's own name so test_ac_traceability.py
    finds NFR-07's evidence where the ledger says to look. Not a new scenario;
    the assertion focus (no exception escapes ainvoke, clean human_review
    route) is NFR-07's own wording, not AC-12's."""
    stub_summarizer(monkeypatch)
    case = build_tool_failure_exhausts_case()
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=case.mcp_tools, model=case.model
    )
    thread = {"configurable": {"thread_id": case.case_id}}
    # No try/except here -- an unhandled exception escaping this call IS the
    # failure this test guards against; pytest would report it as an error,
    # not a clean assertion failure, if graceful degradation ever regressed.
    result = await graph.ainvoke(seed_request_only(case.case_id), config=thread)
    assert "__interrupt__" in result


@pytest.mark.asyncio
async def test_persistent_timeout_ends_at_human_review_not_unhandled_exception(
    memory_store, monkeypatch
):
    """The timeout-specific path, distinct from Task 5's exception-based
    scenario: a tool that just HANGS (simulating a stuck MCP subprocess or a
    slow network call) must still be bounded by worker_timeout_seconds and
    degrade the same way -- reflection.py's run_worker_react_resilient
    converts the asyncio.TimeoutError into a WorkerTimeoutError(tool=
    "_worker_turn_", ...) internally (see reflection.py), which this test
    confirms surfaces correctly all the way through a real graph run.

    Final whole-branch review finding (PR6): a timeout used to be retried by
    tenacity like any other WorkerToolError, pushing worst-case latency to
    max_tool_retries * worker_timeout_seconds per node visit. Fixed by
    excluding WorkerTimeoutError from the retry predicate -- this test no
    longer needs to override PA_MAX_TOOL_RETRIES to stay fast (one script
    entry per node visit is now correct regardless of that setting's value,
    since a timeout is never retried); the test's own wall-clock time (well
    under a second for max_replans node visits at 0.05s each) is itself
    evidence the fix works."""
    monkeypatch.setenv("PA_WORKER_TIMEOUT_SECONDS", "0.05")
    get_settings.cache_clear()  # memory_store fixture already primed the
    # lru_cache with the OLD env value during its own setup, same footgun
    # Task 2's test_intake_tool_failure_sets_needs_replan hit.
    stub_summarizer(monkeypatch)
    settings = get_settings()

    @tool("benefit_lookup")
    async def hanging_benefit_lookup(member_id: str, service_code: str) -> dict:
        """Benefit lookup that hangs forever (simulates a stuck MCP call)."""
        await asyncio.sleep(999)

    model = FakeToolCallingModel(
        script=[
            ai_tool_call("benefit_lookup", {"member_id": MEMBER_ID, "service_code": SERVICE_CODE})
        ]
        * settings.max_replans,
        structured_responses=[],
    )
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=[hanging_benefit_lookup], model=model
    )
    thread = {"configurable": {"thread_id": "case-nfr07-timeout"}}
    result = await graph.ainvoke(seed_request_only("case-nfr07-timeout"), config=thread)

    assert "__interrupt__" in result
    assert result["replan_count"] == settings.max_replans
    assert all(f.tool == "_worker_turn_" for f in result["tool_failures"])
