"""Shared worker infrastructure: the Gemini chat-model constructor, and the
ReAct-loop-with-structured-output helper built on
langgraph.prebuilt.create_react_agent. ToolNode is built with
handle_tool_errors=False so a real tool exception reaches this module, which
converts it into one uniform WorkerToolError for every worker to catch.

Verified empirically (langgraph 1.0.1, installed
langgraph/prebuilt/tool_node.py::_arun_one): with handle_tool_errors=False a
tool's own exception propagates through agent.ainvoke(...) completely bare --
the exact original exception object (same type, same .args, str(exc) ==
str(original)), not wrapped in a langgraph-specific error type, and with no
__cause__/__context__ chaining added. Nothing on the exception itself names
the failing tool -- ToolNode knows the call name locally (`call["name"]`) but
does a bare `raise e` without attaching it. So `_best_effort_tool_name` below
really is best-effort: it can only fall back to "the one tool that was bound"
or "unknown_tool", per the task brief."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import BaseModel

from pa_copilot.config import Settings, get_settings


def get_agent_model(
    *,
    model_name: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> ChatGoogleGenerativeAI:
    s = settings or get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name or s.model_agent,
        temperature=s.temperature_agent if temperature is None else temperature,
    )


class WorkerToolError(Exception):
    def __init__(self, *, tool: str, error: str, attempt: int = 1):
        super().__init__(f"{tool}: {error}")
        self.tool = tool
        self.error = error
        self.attempt = attempt


async def run_worker_react(
    model,
    tools: list[BaseTool],
    *,
    system_prompt: str,
    messages: list[Any],
    response_format: type[BaseModel],
    config: dict | None = None,
    recursion_limit: int = 10,
) -> tuple[list[AnyMessage], BaseModel]:
    agent = create_react_agent(
        model,
        tools=ToolNode(tools, handle_tool_errors=False),
        prompt=system_prompt,
        response_format=response_format,
    )
    run_config = {**(config or {}), "recursion_limit": recursion_limit}
    try:
        result = await agent.ainvoke({"messages": messages}, config=run_config)
    except Exception as exc:  # noqa: BLE001 -- normalize any tool/model failure for callers
        raise WorkerToolError(tool=_best_effort_tool_name(tools, exc), error=str(exc)) from exc
    return result["messages"], result["structured_response"]


def _best_effort_tool_name(tools: list[BaseTool], exc: Exception) -> str:
    # Confirmed empirically (see module docstring): the propagated exception
    # carries no attribute identifying the failing tool, so there is nothing to
    # extract from `exc` itself. The only signal left is the tools list a
    # single worker was built with -- when it's exactly one tool, that's the
    # one that failed; with more than one, PR6's reflection.py is where richer
    # attribution (e.g. threading the call name through ToolNode) belongs.
    return tools[0].name if len(tools) == 1 else "unknown_tool"
