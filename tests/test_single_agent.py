"""single_agent.run_single_agent: one ReAct loop over the union of tools, real
attribution + resilience wrapper reused from PR6 (not reimplemented)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import AIMessage  # noqa: E402

from _fakes import FakeToolCallingModel  # noqa: E402

from pa_copilot.schemas import PADecision  # noqa: E402
from pa_copilot.single_agent import run_single_agent  # noqa: E402


@pytest.mark.asyncio
async def test_single_agent_returns_a_pa_decision(memory_store):
    decision = PADecision(
        disposition="approve", cited_criteria=[], reviewer_summary="ok", confidence=0.8,
    )
    fake = FakeToolCallingModel(
        script=[AIMessage(content="assessed")], structured_responses=[decision]
    )
    result = await run_single_agent(
        "raw provider note", "M100001", mcp_tools=[], store=memory_store, model=fake,
    )
    assert result == decision


@pytest.mark.asyncio
async def test_single_agent_binds_rag_and_memory_tools(memory_store, monkeypatch):
    seen_tools = {}

    async def spy_resilient(model, tools, **kw):
        seen_tools["names"] = {t.name for t in tools}
        return [], PADecision(
            disposition="approve", cited_criteria=[], reviewer_summary="x", confidence=0.5,
        )

    monkeypatch.setattr("pa_copilot.single_agent.run_worker_react_resilient", spy_resilient)
    await run_single_agent("raw", "M1", mcp_tools=[], store=memory_store, model=object())
    assert "search_clinical_guidance" in seen_tools["names"]
    assert "manage_memory" in seen_tools["names"] or any(
        "manage" in n for n in seen_tools["names"]
    )
