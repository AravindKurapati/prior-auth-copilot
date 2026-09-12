"""run_worker_react_resilient: tenacity retry on WorkerToolError, timeout->WorkerToolError,
exactly one lite-model fallback on WorkerOutputError, WorkerRecursionError never retried."""

import asyncio

import pytest

from pa_copilot.agents._react import WorkerOutputError, WorkerRecursionError, WorkerToolError
from pa_copilot.config import get_settings
from pa_copilot.reflection import run_worker_react_resilient


def _settings(**overrides):
    s = get_settings()
    return s.__class__(**{**s.__dict__, "worker_timeout_seconds": 5, "max_tool_retries": 3, **overrides})


@pytest.mark.asyncio
async def test_retries_transient_tool_error_then_succeeds(monkeypatch):
    calls = {"n": 0}

    async def flaky_run_worker_react(*a, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise WorkerToolError(tool="x", error="transient")
        return (["msg"], "structured")

    monkeypatch.setattr("pa_copilot.reflection.run_worker_react", flaky_run_worker_react)
    result = await run_worker_react_resilient(
        model=object(), tools=[], system_prompt="p", messages=[], response_format=str,
        settings=_settings(),
    )
    assert result == (["msg"], "structured")
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_exhausts_retries_and_reraises_worker_tool_error(monkeypatch):
    async def always_fails(*a, **kw):
        raise WorkerToolError(tool="x", error="down")

    monkeypatch.setattr("pa_copilot.reflection.run_worker_react", always_fails)
    with pytest.raises(WorkerToolError):
        await run_worker_react_resilient(
            model=object(), tools=[], system_prompt="p", messages=[], response_format=str,
            settings=_settings(),
        )


@pytest.mark.asyncio
async def test_timeout_becomes_worker_tool_error(monkeypatch):
    async def hangs(*a, **kw):
        await asyncio.sleep(10)

    monkeypatch.setattr("pa_copilot.reflection.run_worker_react", hangs)
    with pytest.raises(WorkerToolError):
        await run_worker_react_resilient(
            model=object(), tools=[], system_prompt="p", messages=[], response_format=str,
            settings=_settings(worker_timeout_seconds=0.05, max_tool_retries=1),
        )


@pytest.mark.asyncio
async def test_output_error_falls_back_to_lite_model_once(monkeypatch):
    seen_models = []

    async def records_model(m, *a, **kw):
        seen_models.append(m)
        if m == "primary":
            raise WorkerOutputError(error="bad schema")
        return (["msg"], "structured-from-lite")

    monkeypatch.setattr("pa_copilot.reflection.run_worker_react", records_model)
    result = await run_worker_react_resilient(
        model="primary", tools=[], system_prompt="p", messages=[], response_format=str,
        settings=_settings(), lite_model="lite",
    )
    assert result == (["msg"], "structured-from-lite")
    assert seen_models == ["primary", "lite"]


@pytest.mark.asyncio
async def test_output_error_without_lite_model_propagates(monkeypatch):
    async def always_bad(*a, **kw):
        raise WorkerOutputError(error="bad schema")

    monkeypatch.setattr("pa_copilot.reflection.run_worker_react", always_bad)
    with pytest.raises(WorkerOutputError):
        await run_worker_react_resilient(
            model="primary", tools=[], system_prompt="p", messages=[], response_format=str,
            settings=_settings(), lite_model=None,
        )


@pytest.mark.asyncio
async def test_recursion_error_never_retried(monkeypatch):
    calls = {"n": 0}

    async def always_recurses(*a, **kw):
        calls["n"] += 1
        raise WorkerRecursionError(error="loop")

    monkeypatch.setattr("pa_copilot.reflection.run_worker_react", always_recurses)
    with pytest.raises(WorkerRecursionError):
        await run_worker_react_resilient(
            model=object(), tools=[], system_prompt="p", messages=[], response_format=str,
            settings=_settings(),
        )
    assert calls["n"] == 1
