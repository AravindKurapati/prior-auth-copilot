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

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import BaseModel, ValidationError

from pa_copilot.config import Settings, get_settings
from pa_copilot.schemas import ToolFailure


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


def get_lite_agent_model(
    *, temperature: float | None = None, settings: Settings | None = None
) -> ChatGoogleGenerativeAI:
    s = settings or get_settings()
    return ChatGoogleGenerativeAI(
        model=s.model_agent_lite,
        temperature=s.temperature_agent if temperature is None else temperature,
    )


class AttributedToolError(Exception):
    """Raised by a tool wrapped via reflection.attribute_tool_errors when THAT tool's
    own execution fails -- carries the real tool name, unlike the bare exception
    ToolNode(handle_tool_errors=False) propagates (see this module's top docstring).
    run_worker_react catches this ahead of its generic except Exception fallback."""

    def __init__(self, *, tool: str, original: Exception):
        super().__init__(f"{tool}: {original}")
        self.tool = tool
        self.original = original


class WorkerToolError(Exception):
    def __init__(self, *, tool: str, error: str, attempt: int = 1):
        super().__init__(f"{tool}: {error}")
        self.tool = tool
        self.error = error
        self.attempt = attempt


class WorkerTimeoutError(WorkerToolError):
    """A worker turn exceeded worker_timeout_seconds -- deliberately a
    WorkerToolError subclass so every worker's existing `except WorkerToolError`
    handler still catches it unchanged, but reflection.run_worker_react_resilient's
    tenacity layer excludes it from retry (final whole-branch review, PR6):
    a blown time budget isn't "transient" the way a flaky tool call is, and
    retrying it 3x before giving up pushed worst-case per-node-visit latency to
    max_tool_retries * worker_timeout_seconds -- an unbounded-feeling delay
    before the supervisor-level reflection loop even gets a turn to react."""


class WorkerOutputError(Exception):
    """The post-loop structured-output call failed validation (or the model
    otherwise couldn't produce a valid response_format instance) -- a
    different remediation path from a tool failure (design.md: retry, not
    re-route-and-blame-a-tool). Raised when create_react_agent's internal
    ``model.with_structured_output(response_format)`` call -- made once the
    tool-calling loop has no more tool calls -- raises pydantic.ValidationError.
    Callers in THIS branch (intake.py, benefit_check.py) don't yet catch this
    distinctly; it propagates uncaught from those two workers until PR6's
    reflection.py builds the retry-vs-reroute logic design.md specifies."""

    def __init__(self, *, error: str):
        super().__init__(error)
        self.error = error


class WorkerRecursionError(Exception):
    """The ReAct loop hit its recursion_limit without the model ever settling
    on a final (no-more-tool-calls) turn -- a distinct failure mode from both
    a single tool's own exception (WorkerToolError) and a structured-output
    validation failure (WorkerOutputError): no specific tool is to blame, and
    nothing was "invalid" -- the loop simply didn't converge in time. Kept
    separate so it is never silently mislabeled as a tool-specific failure;
    remediation semantics (retry with a tighter loop vs. reroute) are PR6's
    reflection.py's concern, not this module's."""

    def __init__(self, *, error: str):
        super().__init__(error)
        self.error = error


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
    except ValidationError as exc:
        # create_react_agent's separate post-loop
        # model.with_structured_output(response_format) call failed to
        # produce a valid instance -- caught BEFORE the broad `except
        # Exception` below so this is never mislabeled as a tool failure
        # (WorkerToolError(tool="unknown_tool", ...)). Must come before
        # GraphRecursionError/Exception since it is unrelated to either.
        raise WorkerOutputError(error=str(exc)) from exc
    except GraphRecursionError as exc:
        # Hit recursion_limit -- not a specific tool's fault either; keep it
        # out of WorkerToolError so callers don't misattribute a bogus tool.
        raise WorkerRecursionError(error=str(exc)) from exc
    except AttributedToolError as exc:
        # A tool wrapped by reflection.attribute_tool_errors failed inside its own
        # execution -- exc.tool is the REAL tool name (not a guess), exc.original the
        # real underlying exception. Normalize to the same WorkerToolError shape every
        # caller already expects.
        raise WorkerToolError(tool=exc.tool, error=str(exc.original), attempt=1) from exc
    except Exception as exc:  # noqa: BLE001 -- last-resort fallback for any UNWRAPPED tool
        raise WorkerToolError(tool=_best_effort_tool_name(tools, exc), error=str(exc)) from exc
    return result["messages"], result["structured_response"]


def tool_failure_update(exc: WorkerToolError) -> dict:
    """The state-update every worker (intake.py, benefit_check.py, ...) builds
    identically from a caught WorkerToolError -- extracted here so there is one
    implementation instead of one copy per worker module."""
    failure = ToolFailure(
        tool=exc.tool,
        error=exc.error,
        attempt=exc.attempt,
        ts=datetime.now(timezone.utc).isoformat(),
    )
    return {"tool_failures": [failure], "needs_replan": True}


def _best_effort_tool_name(tools: list[BaseTool], exc: Exception) -> str:
    # Confirmed empirically (see module docstring): the propagated exception
    # carries no attribute identifying the failing tool, so there is nothing to
    # extract from `exc` itself. The only signal left is the tools list a
    # single worker was built with -- when it's exactly one tool, that's the
    # one that failed; with more than one, PR6's reflection.py is where richer
    # attribution (e.g. threading the call name through ToolNode) belongs.
    return tools[0].name if len(tools) == 1 else "unknown_tool"
