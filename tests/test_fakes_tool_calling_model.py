"""FakeToolCallingModel: the one test double every PR5(a/b) worker test needs —
scripted tool-calling turns + a scripted final structured-output object, driven
through the real langgraph.prebuilt.create_react_agent (not a hand-rolled loop)."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from _fakes import FakeToolCallingModel, ai_tool_call


class Out(BaseModel):
    value: str


@tool
def echo(x: str) -> str:
    """Echo x."""
    return f"echoed:{x}"


@pytest.mark.asyncio
async def test_tool_loop_then_structured_output():
    model = FakeToolCallingModel(
        script=[
            ai_tool_call("echo", {"x": "hi"}),
            AIMessage(content="done"),
        ],
        structured_responses=[Out(value="final")],
    )
    agent = create_react_agent(model, tools=[echo], response_format=Out)
    result = await agent.ainvoke({"messages": [("user", "go")]})
    assert result["structured_response"] == Out(value="final")
    kinds = [type(m).__name__ for m in result["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]


def test_script_exhaustion_raises_clear_error():
    model = FakeToolCallingModel(script=[AIMessage(content="only one")])
    model.invoke([("user", "a")])
    with pytest.raises(AssertionError, match="script exhausted"):
        model.invoke([("user", "b")])
