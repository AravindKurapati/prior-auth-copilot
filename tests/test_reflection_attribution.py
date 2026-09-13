"""AttributedToolError: real per-tool attribution, replacing the "unknown_tool" guess
(PR5a/5b's carried-forward gap)."""

import asyncio

import pytest
from langchain_core.tools import StructuredTool

from pa_copilot.agents._react import AttributedToolError
from pa_copilot.reflection import attribute_tool_errors
from pa_copilot.rag.tool import search_clinical_guidance


def test_wraps_sync_tool_and_attributes_failure():
    def boom(*a, **kw):
        raise RuntimeError("index down")

    # Create a throwaway copy of the tool to avoid mutating the shared production singleton.
    # model_copy() is available on Pydantic-backed StructuredTool instances.
    tool_copy = search_clinical_guidance.model_copy(update={"func": boom})
    (wrapped,) = attribute_tool_errors([tool_copy])
    with pytest.raises(AttributedToolError) as exc_info:
        wrapped.invoke({"query": "x"})
    assert exc_info.value.tool == "search_clinical_guidance"
    assert "index down" in str(exc_info.value.original)


def test_wraps_async_tool_and_attributes_failure():
    async def boom(**kw):
        raise RuntimeError("mcp down")

    tool = StructuredTool.from_function(
        coroutine=boom, name="criteria_check", description="fake"
    )
    (wrapped,) = attribute_tool_errors([tool])
    with pytest.raises(AttributedToolError) as exc_info:
        asyncio.run(wrapped.ainvoke({}))
    assert exc_info.value.tool == "criteria_check"


def test_idempotent_double_wrap_does_not_nest_attribution():
    async def boom(**kw):
        raise RuntimeError("mcp down")

    tool = StructuredTool.from_function(coroutine=boom, name="provider_lookup", description="fake")
    attribute_tool_errors([tool])
    attribute_tool_errors([tool])  # simulates the SAME shared mcp_tools object being
                                   # wrapped by more than one worker builder
    with pytest.raises(AttributedToolError) as exc_info:
        asyncio.run(tool.ainvoke({}))
    assert exc_info.value.tool == "provider_lookup"
    assert isinstance(exc_info.value.original, RuntimeError)  # NOT a nested AttributedToolError


def test_success_path_unaffected():
    async def ok(**kw):
        return "fine"

    tool = StructuredTool.from_function(coroutine=ok, name="ok_tool", description="fake")
    (wrapped,) = attribute_tool_errors([tool])
    assert asyncio.run(wrapped.ainvoke({})) == "fine"
