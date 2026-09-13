"""PR8 good-to-have: the background importance-weighted memory manager
(design.md §6.1's own citation: "create_memory_store_manager for background
importance-weighted extraction"). Runs outside the synchronous agent loop.

`create_memory_store_manager` itself is langmem's own well-tested surface --
these tests verify OUR wiring (namespace, store, instructions, invocation
shape), not langmem's internal extraction behavior, by stubbing the manager
langmem hands back."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from pa_copilot.memory.manager import (
    EPISODIC_NAMESPACE,
    build_case_memory_manager,
    enrich_case_memory,
)


def test_episodic_namespace_matches_design_doc():
    assert EPISODIC_NAMESPACE == ("pa", "episodic")


def test_build_case_memory_manager_wires_store_and_namespace(monkeypatch, memory_store):
    captured = {}

    def fake_create_memory_store_manager(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return "the-manager"

    import pa_copilot.memory.manager as mgr

    monkeypatch.setattr(mgr, "create_memory_store_manager", fake_create_memory_store_manager)

    manager = build_case_memory_manager(memory_store, model="fake-model")

    assert manager == "the-manager"
    assert captured["model"] == "fake-model"
    assert captured["namespace"] == ("pa", "episodic")
    assert captured["store"] is memory_store
    assert "critical" in captured["instructions"]
    assert captured["enable_deletes"] is True


@pytest.mark.asyncio
async def test_enrich_case_memory_invokes_manager_with_messages():
    calls = []

    class FakeManager:
        async def ainvoke(self, payload):
            calls.append(payload)
            return ["updated-item"]

    messages = [HumanMessage(content="case transcript")]
    result = await enrich_case_memory(FakeManager(), messages)

    assert result == ["updated-item"]
    assert calls == [{"messages": messages}]
