"""SummarizationNode wrapper: verified library contract is `input["context"]` in,
`{"summarized_messages": ..., "context": {"running_summary": ...}}` out (only
when a summary actually fires) -- confirmed against the installed
langmem.short_term.summarization source (`SummarizationNode._prepare_state_update`).

Note: `_preprocess_messages` (the library's internal helper) raises
`ValueError("Messages are required to have ID field.")` for *any* message with
`.id is None`, regardless of whether the token threshold is actually crossed --
so every message built here carries an explicit `id=`, unlike the raw
`HumanMessage(content=...)` sketch in the task brief (verified empirically: a
plain `HumanMessage(content="short")` with no id raises before any threshold
logic runs). Also empirically verified: `max_summary_tokens` must be strictly
less than `max_tokens` (a library invariant not mentioned in the brief's test
sketch), which matters once a test picks a small `max_tokens`."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from _fakes import FakeToolCallingModel
from pa_copilot.context.summarization import build_summarization_node, summarize


def _long_thread(n: int) -> list:
    msgs = []
    for i in range(n):
        msgs.append(
            HumanMessage(
                id=f"h{i}",
                content=f"Provider note {i}: patient continues PT, no change." * 5,
            )
        )
        msgs.append(AIMessage(id=f"a{i}", content=f"Acknowledged note {i}."))
    return msgs


@pytest.mark.asyncio
async def test_summarize_under_threshold_no_summary_yet():
    node = build_summarization_node(
        model=FakeToolCallingModel(script=[]),
        max_tokens=100_000,
        max_tokens_before_summary=100_000,
    )
    result = await node.ainvoke(
        {"messages": [HumanMessage(id="h0", content="short")], "context": {}}
    )
    assert result["summarized_messages"]
    assert "context" not in result  # threshold not crossed -> no summary produced


@pytest.mark.asyncio
async def test_summarize_over_threshold_produces_running_summary():
    # `max_summary_tokens` must be < `max_tokens` (library invariant, verified
    # empirically -- the default 256 fails against this test's small
    # max_tokens=200), so it's pinned below both other thresholds here.
    fake = FakeToolCallingModel(script=[AIMessage(content="Summary: patient on ongoing PT course.")])
    node = build_summarization_node(
        model=fake, max_tokens=200, max_tokens_before_summary=50, max_summary_tokens=20
    )
    result = await node.ainvoke({"messages": _long_thread(20), "context": {}})
    assert "context" in result
    assert result["context"]["running_summary"] is not None
    assert len(result["summarized_messages"]) < 40  # compressed vs. the 40 raw messages


@pytest.mark.asyncio
async def test_summarize_node_wrapper_is_graph_node_shaped(monkeypatch):
    """Exercises the `summarize()` wrapper itself (not just build_summarization_node).

    `summarize()` caches its node at module level (`_node`) and, on first use,
    builds it via `_default_model()` -- a real `ChatGoogleGenerativeAI`, which
    (verified empirically) raises `DefaultCredentialsError` at *construction*
    time when no Google/Gemini credentials are configured, well before any
    network call. To keep this test hermetic and independent of test ordering,
    pre-seed the module-level cache with a fake-backed node so `_default_model()`
    is never reached; monkeypatch restores `_node` automatically afterward."""
    import pa_copilot.context.summarization as summarization_mod

    fake_node = build_summarization_node(
        model=FakeToolCallingModel(script=[]),
        max_tokens=100_000,
        max_tokens_before_summary=100_000,
    )
    monkeypatch.setattr(summarization_mod, "_node", fake_node)

    state = {"messages": [HumanMessage(id="h0", content="hi")], "context": {}}
    update = await summarize(state)
    assert "summarized_messages" in update
