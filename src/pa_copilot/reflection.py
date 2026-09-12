"""Reflection / self-healing infrastructure (design.md §3.5, §3.6; AC-12, NFR-07).

Two independent mechanisms:

1. attribute_tool_errors -- wraps each tool's .func/.coroutine in place so a failure
   inside THAT tool's own execution is tagged with its real name before ToolNode's bare
   `raise e` (see agents/_react.py's top docstring) ever discards attribution. Idempotent
   (safe to call more than once on the same tool objects -- the MCP tools list is shared
   across three worker builders, see graph.py's own docstring) via a `_pa_attributed`
   marker, verified settable on a StructuredTool instance despite extra="ignore".

2. run_worker_react_resilient -- wraps run_worker_react with a per-turn asyncio.wait_for
   timeout, tenacity exponential-backoff retries (same model) on a transient
   WorkerToolError, and exactly one additional attempt on a lite model when the
   structured-output call itself failed validation (WorkerOutputError) -- a DISTINCT
   failure mode from a tool error, per design.md's NFR-07 row separating "transient tool
   errors" from "model-call failure". WorkerRecursionError is never retried by either
   mechanism (see this plan's Design decisions).
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from pa_copilot.agents._react import (
    AttributedToolError,
    WorkerOutputError,
    WorkerToolError,
    run_worker_react,
)
from pa_copilot.config import Settings, get_settings


def attribute_tool_errors(tools: list[BaseTool]) -> list[BaseTool]:
    for tool in tools:
        if getattr(tool, "_pa_attributed", False):
            continue
        name = tool.name
        if getattr(tool, "coroutine", None) is not None:
            original_coro = tool.coroutine

            async def _wrapped_coro(*args: Any, _name: str = name, _orig=original_coro, **kw: Any):
                try:
                    return await _orig(*args, **kw)
                except Exception as exc:  # noqa: BLE001 -- deliberately broad, re-tagged below
                    raise AttributedToolError(tool=_name, original=exc) from exc

            tool.coroutine = _wrapped_coro
        if getattr(tool, "func", None) is not None:
            original_func = tool.func

            def _wrapped_func(*args: Any, _name: str = name, _orig=original_func, **kw: Any):
                try:
                    return _orig(*args, **kw)
                except Exception as exc:  # noqa: BLE001
                    raise AttributedToolError(tool=_name, original=exc) from exc

            tool.func = _wrapped_func
        tool._pa_attributed = True
    return tools


async def run_worker_react_resilient(
    model,
    tools: list[BaseTool],
    *,
    system_prompt: str,
    messages: list[Any],
    response_format: type[BaseModel],
    config: dict | None = None,
    recursion_limit: int = 10,
    settings: Settings | None = None,
    lite_model=None,
):
    s = settings or get_settings()

    async def _attempt(m):
        try:
            return await asyncio.wait_for(
                run_worker_react(
                    m,
                    tools,
                    system_prompt=system_prompt,
                    messages=messages,
                    response_format=response_format,
                    config=config,
                    recursion_limit=recursion_limit,
                ),
                timeout=s.worker_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise WorkerToolError(
                tool="_worker_turn_", error=f"exceeded {s.worker_timeout_seconds}s"
            ) from exc

    retrying = AsyncRetrying(
        stop=stop_after_attempt(s.max_tool_retries),
        wait=wait_exponential(multiplier=0.5, max=5),
        retry=retry_if_exception_type(WorkerToolError),
        reraise=True,
    )
    try:
        result = None
        async for attempt in retrying:
            with attempt:
                result = await _attempt(model)
        return result
    except WorkerOutputError:
        if lite_model is None:
            raise
        return await _attempt(lite_model)
