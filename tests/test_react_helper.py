"""agents/_react.py: the shared ReAct-loop-with-structured-output helper every
PR5(a/b) worker builds on. Verifies (a) the happy path through
create_react_agent + FakeToolCallingModel, (b) a tool exception surfaces as
WorkerToolError rather than propagating raw or being silently swallowed, (c) a
pydantic.ValidationError from the separate post-loop structured-output call
surfaces as WorkerOutputError, NOT WorkerToolError -- a validation failure has
nothing to do with any tool and must not be blamed on one."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import BaseModel, ValidationError

from _fakes import FakeToolCallingModel, ai_tool_call
from pa_copilot.agents._react import (
    WorkerOutputError,
    WorkerToolError,
    get_agent_model,
    run_worker_react,
)


class Out(BaseModel):
    value: str


@tool
def good_tool(x: str) -> str:
    """A tool that works."""
    return f"ok:{x}"


@tool
def bad_tool(x: str) -> str:
    """A tool that always raises."""
    raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_happy_path_returns_messages_and_structured_output():
    model = FakeToolCallingModel(
        script=[ai_tool_call("good_tool", {"x": "hi"}), AIMessage(content="done")],
        structured_responses=[Out(value="final")],
    )
    messages, structured = await run_worker_react(
        model, [good_tool], system_prompt="You are a test worker.",
        messages=[("user", "go")], response_format=Out,
    )
    assert structured == Out(value="final")
    assert any(getattr(m, "content", None) == "ok:hi" for m in messages)


@pytest.mark.asyncio
async def test_tool_exception_becomes_worker_tool_error():
    model = FakeToolCallingModel(
        script=[ai_tool_call("bad_tool", {"x": "hi"})],
        structured_responses=[Out(value="unreachable")],
    )
    with pytest.raises(WorkerToolError) as exc_info:
        await run_worker_react(
            model, [bad_tool], system_prompt="You are a test worker.",
            messages=[("user", "go")], response_format=Out,
        )
    assert "boom" in exc_info.value.error
    assert exc_info.value.attempt == 1


@pytest.mark.asyncio
async def test_structured_output_validation_error_becomes_worker_output_error():
    """create_react_agent's post-loop model.with_structured_output(response_format)
    call is separate from the tool-calling loop -- a validation failure there
    is not a tool failure. Must raise WorkerOutputError, never
    WorkerToolError(tool="unknown_tool", ...)."""

    class InvalidStructuredOutputModel(FakeToolCallingModel):
        def with_structured_output(self, schema, **kwargs):
            def _raise(*_args, **_kwargs):
                raise ValidationError.from_exception_data(schema.__name__, [])

            return RunnableLambda(_raise)

    model = InvalidStructuredOutputModel(script=[AIMessage(content="done")])
    with pytest.raises(WorkerOutputError) as exc_info:
        await run_worker_react(
            model, [good_tool], system_prompt="You are a test worker.",
            messages=[("user", "go")], response_format=Out,
        )
    assert "validation error" in exc_info.value.error.lower()


def test_get_agent_model_uses_settings_defaults(monkeypatch):
    # ChatGoogleGenerativeAI's constructor resolves credentials eagerly (verified
    # empirically, matching test_context_summarization.py's documented finding):
    # with no GOOGLE_API_KEY it falls through to google.auth.default() and raises
    # DefaultCredentialsError in an environment with no ADC configured. An API
    # key (any string) short-circuits that lookup without making a network call,
    # so this stays hermetic like every other model-construction test here (see
    # tests/test_config.py's GEMINI_API_KEY monkeypatch pattern).
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")
    model = get_agent_model()
    assert model.model.endswith("gemini-flash-latest") or "gemini-flash-latest" in model.model
