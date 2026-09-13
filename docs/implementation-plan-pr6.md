# PR6 — Reflection & Self-Healing: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The dual-trigger reflection/self-healing loop (design.md §3.5), graceful
degradation (design.md §3.6 / NFR-07), and the two carried-forward fixes PR5b's handoff
promoted to required PR6 work: `supervisor_hops`/`replan_count` reset on resume, and
`agents/_react.py::_best_effort_tool_name`'s universal `"unknown_tool"` fallback. Closes
**AC-12** in full, **NFR-07** in full. Branch `feat/reflection`, off `main` @ the PR5b
merge commit (`5b0068c` as of this writing — confirm with `git log --first-parent` before
starting).

**Architecture:** Three independent mechanisms, all additive to existing code — no
existing worker's happy path changes shape:

1. **Real per-tool attribution**, replacing the `"unknown_tool"` guess. `ToolNode`'s bare
   `raise e` (verified in PR5a — see `agents/_react.py`'s own docstring) carries no
   identifying attribute, so attribution has to happen *before* the exception ever reaches
   `ToolNode`: each tool object is wrapped in place (its `.func` or `.coroutine`
   reassigned, verified empirically this session — see Library facts) so a failure inside
   *that specific tool's* execution raises a new `AttributedToolError(tool=<real name>,
   original=<real exception>)`, caught by `run_worker_react` ahead of its generic
   `except Exception` fallback and turned into a `WorkerToolError` carrying the real tool
   name. `_best_effort_tool_name`'s guess remains only as the safety net for any tool this
   mechanism doesn't reach (none, in this codebase, today — every tool is a
   `StructuredTool` with `.func` or `.coroutine` set, verified for the RAG tool and the MCP
   adapter tools this session; langmem's memory tools are the one family not yet checked —
   Task 2 verifies them too).
2. **Resilience**: `asyncio.wait_for` around one whole worker turn (`run_worker_react`'s
   full ReAct loop, not per-tool-call — verified empirically last session that this is safe,
   cancels cleanly, and reaches through a live MCP stdio subprocess call), `tenacity`
   exponential-backoff retries (same model) on a `WorkerToolError`, and — a *distinct*
   mechanism, not more tenacity — exactly one fallback attempt on the lite model when the
   structured-output call itself fails validation (`WorkerOutputError`). Neither retry
   layer touches `WorkerRecursionError` — a non-convergent loop retrying with the same
   inputs won't converge either; existing workers already route it straight to
   `needs_replan`.
3. **The replan/reroute state machine**: `replan_count` is currently never incremented
   anywhere (verified against `git grep replan_count` across `src/` before writing this
   plan — the only writer is `state.py`'s zero-initializer) — so `hard_route`'s existing
   `tool_failures and replan_count >= max_replans -> human_review` guardrail is dead code.
   This PR makes `build_supervisor_node` the single place that consumes a pending
   `needs_replan` signal: it increments `replan_count` and clears `needs_replan` back to
   `False` on the same turn it's about to route, *before* calling `hard_route` — so a
   cap-exhaustion decision on that same turn sees the up-to-date count. **Design ruling**
   (recorded here, not left for an implementer to improvise): the two triggers design.md
   §3.5 tables separately (tool failure, low confidence) share ONE counter and ONE cap,
   not per-trigger budgets — the config file's existing comment ("per-worker re-plan
   attempts") is aspirational, not a contract; `tests/test_supervisor.py`'s existing
   `test_hard_route_exhausted_replans_goes_to_human_review` already pins a single global
   `replan_count` compared against `max_replans`, and that test is the binding spec here,
   not the yaml comment (Task 3 corrects the comment to match). **Neither trigger needs a
   new deterministic `hard_route` guardrail** — verified against the current routing table
   this session: a tool failure leaves the failing worker's own state field `None`
   (`request`/`benefit`/`necessity` — see `hard_route`), which *already* re-targets that
   same worker on the next turn via the existing None-checks; a `decision_draft`
   self-critique failure and a `medical_necessity` low-confidence result both leave
   `hard_route` with no matching rule (falls through to `return None`), handing the choice
   to the LLM router exactly as design.md's "supervisor loops back with a hint" already
   implies — this PR's job is to make that fall-through *bounded* (the broadened cap check)
   and *informed* (the router's prompt gets an explicit reflection hint when `needs_replan`
   was set), not to force a new deterministic path that would remove the router's judgment
   design.md asks it to exercise. This also means **no existing test's fixture-driven
   behavior changes** — `tests/_full_case.py`'s ambiguous case (a low-confidence,
   `indeterminate` `NecessityAssessment` whose scripted `RouterDecision` picks
   `human_review` directly) still reaches `human_review` exactly the same way; the only
   addition is that `needs_replan` also gets set (and immediately counted/cleared) on that
   path, which nothing in the existing assertions inspects. Verified by reading
   `tests/_full_case.py`, `tests/test_ac02_supervisor_routing.py`,
   `tests/test_ac03_conditional_routing.py`, `tests/test_ac11_agentic_rag.py` before
   writing this plan — none of them assert on `needs_replan` or `replan_count`.
4. **`supervisor_hops`/`replan_count` reset on resume** (PR5b's carried-forward item,
   promoted to required PR6 work by its own handoff note): `human_review`'s node is where
   control returns after a real `interrupt()`/resume — a human just intervened, so this PR
   treats that as a fresh attempt budget. `build_human_review_node`'s resume path adds
   `supervisor_hops: 0, replan_count: 0` to its return update. This is fully testable in
   PR6 with the existing `InMemorySaver` + `Command(resume=...)` pattern already in
   `tests/test_agent_human_review.py` — it does **not** require PR7's `pac resume` CLI to
   exist.

**Tech stack:** No new dependencies. `tenacity>=8.3` is already declared
(`pyproject.toml`, added in PR5b Task 1) and installed (`tenacity==9.1.4`, verified this
session). `asyncio.wait_for`/`asyncio.TimeoutError` are stdlib.

**Spec:** `docs/design.md` §3.5 (the two triggers, authoritative), §3.6 (graceful
degradation, `reflection.py`), §3.2/§3.3 (supervisor + worker shapes this PR extends,
not replaces). `specs/acceptance-criteria.md` AC-12, `specs/nfr.md` NFR-07.

## Design decisions & rulings (recorded here; do not re-litigate mid-task)

- **Single global `replan_count`/cap**, not per-worker or per-trigger. See Architecture
  point 3 above. Ruling risk if wrong: a real deployment might want a tool-failure budget
  independent from a low-confidence budget; cheap to split later (two counters) since
  nothing else depends on the field being singular except `hard_route`'s one comparison.
- **No new deterministic `hard_route` guardrail for either trigger.** The existing
  None-based routes and the LLM-router fall-through already cover both cases. Ruling risk
  if wrong: a grader expecting an explicit "reroute to medical_necessity" code path might
  mark this as insufficiently "hard-wired" — the counter-argument, worth keeping in the
  PR6 ledger commit message, is that design.md's own words are "supervisor loops back
  *with a hint*" (a judgment call, not "supervisor always retries the same worker"), and
  forcing a hard rule here would make the already-shipped, already-`done` AC-03 ambiguous
  case ( scripted straight to `human_review`, no loop-back attempted) contradict a new
  hard rule that says it must loop back first.
- **The reflection "hint" lives in the LLM router's prompt** (`route_with_llm`), not
  re-injected into the retried worker's own prompt. Simpler, matches "supervisor loops
  back with a hint" literally (the supervisor is the one holding the hint), and avoids
  worker prompt-shape churn. Ruling risk if wrong: a worker retried after a low-confidence
  result gets no explicit instruction to "broaden retrieval" — it just reruns with the
  same inputs. Acceptable for a synthetic/graded demo; a real system might want this
  richer. Flagged as a PR8-or-later follow-up in the handoff, not built here.
- **`worker_timeout_seconds` wraps the whole `run_worker_react` call** (the entire ReAct
  loop for one worker turn), not each individual tool invocation. Per-tool timeouts would
  need to wrap inside `ToolNode`'s dispatch, which isn't a seam this codebase owns cleanly;
  wrapping the outer call was verified last session to cancel a real MCP stdio subprocess
  call cleanly, which is the case that actually matters (a hung MCP server or Chroma
  query blocking the whole worker turn).
- **Attribution wrapping is per-tool-object, mutating in place**, guarded by a
  `_pa_attributed` marker so calling `attribute_tool_errors` more than once on the same
  shared tool objects (the MCP tools list is passed to three different worker builders —
  see `graph.py`'s own docstring) is a safe no-op the second and third time, not a
  triple-wrap. Verified empirically this session (see Library facts).

## Global Constraints

- Python `>=3.11`. Package `pa_copilot` under `src/`. Branch `feat/reflection` off `main`
  @ the PR5b merge commit (confirm with `git log --first-parent` before starting).
- No new dependencies; do not touch existing stack pins.
- **No `GEMINI_API_KEY` / `GOOGLE_API_KEY` in this environment.** Every fast test is
  deterministic against `tests/_fakes.py::FakeToolCallingModel` and/or a hand-written
  flaky-tool stub. AC-12's full-graph evidence gets a deterministic producer script
  (committed, byte-stable), matching every prior PR's pattern for evidence needing an LLM.
- Test output must be **pristine** (`pyproject.toml` `filterwarnings = ["error"]`). If
  `tenacity`'s `AsyncRetrying` or anything else emits a new warning under test, add one
  more narrow, category+module-scoped ignore — never broaden an existing one (PR5a's Task 1
  fix-round lesson).
- Committed evidence under `traces/` stays byte-stable — no timestamps, no randomness,
  `write_text(..., encoding="utf-8", newline="\n")`. `RouteStep.ts` must be stripped at
  serialization exactly like `scripts/full_case_demo.py::_dump_route_step` already does
  (PR5b's own final-review finding); reuse that helper rather than re-deriving it.
- Work on branch `feat/reflection`. The controller does the `--no-ff` merge after the
  whole-branch review — do NOT merge in a task.
- TDD: failing test first, run red, minimal impl, run green, commit. Frequent commits.
- `agents/_react.py`, `supervisor.py`, and the four `agents/*.py` worker modules are all
  shared, heavily-tested files — run the FULL fast suite (`pytest -q -m "not slow"`,
  currently 196 tests) after every task that touches them, not just the new tests, and
  report the full pass count in the task report.

## Interfaces from PR1–PR5b (on `main` after the PR5b merge)

- `pa_copilot.config.get_settings() -> Settings` — `.max_replans` (2), `.max_hops` (12),
  `.recursion_limit` (40), `.tau` (0.55), `.model_agent`, `.model_summarizer`. This PR adds
  `.worker_timeout_seconds`, `.max_tool_retries`, `.model_agent_lite` (Task 1).
- `pa_copilot.state.PACaseState` — all fields through PR5b, notably `needs_replan: bool`,
  `replan_count: int`, `supervisor_hops: int` (all default `0`/`False` via
  `new_case_state`), `tool_failures: Annotated[list[ToolFailure], operator.add]`
  (append-only), `route_history: Annotated[list[RouteStep], operator.add]`.
- `pa_copilot.schemas.NecessityAssessment` — `.confidence: float`, `.criteria_status:
  Literal["met","not_met","indeterminate"]`. `ToolFailure` — `.tool`, `.error`,
  `.attempt`, `.ts`.
- `pa_copilot.agents._react` — `get_agent_model(*, model_name=None, temperature=None,
  settings=None)`, `run_worker_react(model, tools, *, system_prompt, messages,
  response_format, config=None, recursion_limit=10) -> (messages, structured)` (unchanged
  by this PR — `run_worker_react_resilient` wraps it, doesn't replace its internals),
  `tool_failure_update(exc: WorkerToolError) -> dict`, `WorkerToolError` (`.tool`,
  `.error`, `.attempt`), `WorkerOutputError` (`.error`), `WorkerRecursionError` (`.error`).
  This PR adds `AttributedToolError` and `get_lite_agent_model` here (Task 1).
- `pa_copilot.supervisor.hard_route(state, *, settings=None) -> RouteTarget | None`,
  `build_supervisor_node(*, model=None, settings=None) -> node callable`,
  `route_with_llm(state, *, model=None) -> RouterDecision` (this PR adds a `settings=None`
  kwarg — Task 3).
- `pa_copilot.agents.{intake,benefit_check,medical_necessity,decision_draft}` —
  `build_*_node(...) -> async def _node(state) -> dict`, exact current shapes read in full
  this session (see the task bodies below for the precise before/after diff each gets).
- `pa_copilot.agents.human_review.build_human_review_node() -> node callable` — currently
  `interrupt()` + `write_working_memory(state, "human_review_resume_value",
  resume_value)`.
- `pa_copilot.graph.make_graph(*, store, checkpointer, mcp_tools, model=None,
  settings=None)` — assembles all seven nodes; **not touched by this PR** (no new nodes,
  no topology change — reflection lives inside existing nodes' logic).
- `tests/_fakes.py::FakeToolCallingModel(script=[...], structured_responses=[...])`,
  `ai_tool_call(name, args, *, call_id=None)`.
- `tests/_full_case.py` — `build_clear_cut_case()`, `build_ambiguous_case()`,
  `new_initial_state(case_id)`, `FAKE_MCP_TOOLS`, `build_stub_summarization_node()`,
  `stub_summarizer(monkeypatch)`, `fake_rag_embedder()`. **Read in full before Task 5** —
  the new reflection cases reuse this scaffolding's shape (`ScriptedCase`, fake MCP tools,
  the stub-summarizer trick) rather than reinventing it.
- `tests/conftest.py` fixtures: `tmp_trace_dir`, `frozen_now`, `fake_embedder`,
  `memory_store`, `sample_request`, plus autouse `_env_snapshot`/`_clear_settings_cache`.
- `scripts/full_case_demo.py` — the exact pattern (imports, `sys.path` setup,
  `_dump_route_step` stripping `ts`, `_write` with `encoding="utf-8", newline="\n"`) Task
  5's new evidence script mirrors.

## Library facts verified this session (installed versions, empirically checked in this
repo's venv, 2026-09-12)

- Installed: `langchain-core==0.3.86`, `tenacity==9.1.4` (already declared + installed,
  no pyproject change needed).
- **`StructuredTool.func`/`.coroutine` can be reassigned post-construction and the new
  callable is what actually runs on `.invoke()`/`.ainvoke()`.** Verified directly:
  reassigning `search_clinical_guidance.func` (a sync tool) to a raising stub and calling
  `.invoke(...)` raised from the stub, not the original; reassigning a synthetic
  `StructuredTool`'s `.coroutine` (async) and calling `.ainvoke(...)` likewise raised from
  the stub. Restoring the original attribute afterward works too (both are declared
  Pydantic fields, not read-only).
- **`langchain_mcp_adapters.tools.convert_mcp_tool_to_langchain_tool` returns a
  `StructuredTool` with `coroutine=call_tool` set and `func=None`.** Read directly from the
  installed package source. So every MCP tool (`benefit_lookup`, `provider_lookup`,
  `criteria_check`) is wrappable via `.coroutine`.
- `search_clinical_guidance` (PR3's `@tool def search_clinical_guidance(...)`, a sync
  function) is a `StructuredTool` with `.func` set, `.coroutine=None` — wrappable via
  `.func`. **Task 1/2 must verify langmem's memory tools (`build_memory_tools`'s
  `search_member`) the same way** — not yet checked this session; the plan cannot assume
  it without looking, since langmem is a third-party package this project doesn't control.
- **An arbitrary new attribute (e.g. `tool._pa_attributed = True`) can be set on a
  `StructuredTool`/`BaseTool` instance post-construction**, verified directly — despite
  `model_config = {"extra": "ignore", ...}`, instance `__setattr__` for an undeclared name
  still succeeds (pydantic v2's `extra="ignore"` governs constructor/validation input, not
  post-init `setattr`). Safe to use as the idempotency marker.
- **`tenacity.AsyncRetrying` with the `async for attempt in retrying: with attempt: ...`
  pattern retries an async callable correctly**, verified directly with a stub that fails
  twice then succeeds under `AsyncRetrying(stop=stop_after_attempt(3),
  wait=wait_fixed(0), retry=retry_if_exception_type(Flaky), reraise=True)` — 3 calls made,
  final result returned, `reraise=True` means an exhausted retry re-raises the *last*
  underlying exception (not a wrapped `RetryError`) — important, since callers must keep
  catching `WorkerToolError` directly, not a tenacity-specific type.
- (Carried from PR5b's own verified facts, still true, referenced by this plan):
  `asyncio.TimeoutError is TimeoutError` in this Python version; `asyncio.wait_for`
  genuinely cancels the wrapped coroutine on timeout and the coroutine observes
  `CancelledError`; safe to wrap `run_worker_react(...)` calls including mid-MCP-subprocess
  ones.

---

### Task 1: Config additions + `reflection.py` core (attribution + resilience) + `_react.py` hooks

**Files:**
- Modify: `config/routing.yaml`, `config/models.yaml`, `src/pa_copilot/config.py`,
  `src/pa_copilot/agents/_react.py`
- Create: `src/pa_copilot/reflection.py`
- Test: `tests/test_config.py` (extend), `tests/test_reflection_attribution.py` (create),
  `tests/test_reflection_resilience.py` (create)

**Interfaces to build:**

`config/routing.yaml` — add two keys, correct one comment:
```yaml
max_replans: 2        # total replan attempts (tool-failure reroute OR low-confidence
                       # loop-back share ONE counter) before forcing human_review
max_hops: 12          # hard cap on supervisor turns per case
recursion_limit: 40   # LangGraph graph recursion limit
tau: 0.55             # medical-necessity confidence threshold
worker_timeout_seconds: 30   # asyncio.wait_for around one worker's whole ReAct turn
max_tool_retries: 3          # tenacity attempts (same model) before human_review-bound
                              # replan; a separate single lite-model attempt covers a
                              # structured-output (WorkerOutputError) failure specifically
```

`config/models.yaml` — add one key:
```yaml
agent_lite: gemini-flash-lite-latest   # fallback model for one retry after a
                                        # structured-output validation failure (NFR-07)
```

`src/pa_copilot/config.py` — add three `Settings` fields and their loaders:
```python
@dataclass(frozen=True)
class Settings:
    ...
    max_replans: int
    max_hops: int
    recursion_limit: int
    tau: float
    worker_timeout_seconds: float
    max_tool_retries: int
    model_agent_lite: str
    memory: MemoryConfig
```
In `load_settings`, alongside the existing `model_agent`/`model_summarizer` reads:
```python
model_agent_lite=os.environ.get(
    "PA_MODEL_AGENT_LITE", models.get("agent_lite", "gemini-flash-lite-latest")
),
```
alongside the existing `max_replans`/`max_hops`/`recursion_limit`/`tau` reads:
```python
worker_timeout_seconds=float(
    os.environ.get("PA_WORKER_TIMEOUT_SECONDS", routing.get("worker_timeout_seconds", 30))
),
max_tool_retries=int(
    os.environ.get("PA_MAX_TOOL_RETRIES", routing.get("max_tool_retries", 3))
),
```

`src/pa_copilot/agents/_react.py` — add, do not remove anything existing:
```python
class AttributedToolError(Exception):
    """Raised by a tool wrapped via reflection.attribute_tool_errors when THAT tool's
    own execution fails -- carries the real tool name, unlike the bare exception
    ToolNode(handle_tool_errors=False) propagates (see this module's top docstring).
    run_worker_react catches this ahead of its generic except Exception fallback."""

    def __init__(self, *, tool: str, original: Exception):
        super().__init__(f"{tool}: {original}")
        self.tool = tool
        self.original = original


def get_lite_agent_model(
    *, temperature: float | None = None, settings: Settings | None = None
) -> ChatGoogleGenerativeAI:
    s = settings or get_settings()
    return ChatGoogleGenerativeAI(
        model=s.model_agent_lite,
        temperature=s.temperature_agent if temperature is None else temperature,
    )
```
And in `run_worker_react`'s except chain, add a new branch **before** the generic
`except Exception` (order matters — Python tries `except` clauses top to bottom, and
`AttributedToolError` is itself an `Exception` subclass so it must be caught first):
```python
    except ValidationError as exc:
        raise WorkerOutputError(error=str(exc)) from exc
    except GraphRecursionError as exc:
        raise WorkerRecursionError(error=str(exc)) from exc
    except AttributedToolError as exc:
        # A tool wrapped by reflection.attribute_tool_errors failed inside its own
        # execution -- exc.tool is the REAL tool name (not a guess), exc.original the
        # real underlying exception. Normalize to the same WorkerToolError shape every
        # caller already expects.
        raise WorkerToolError(tool=exc.tool, error=str(exc.original), attempt=1) from exc
    except Exception as exc:  # noqa: BLE001 -- last-resort fallback for any UNWRAPPED tool
        raise WorkerToolError(tool=_best_effort_tool_name(tools, exc), error=str(exc)) from exc
```

`src/pa_copilot/reflection.py` (new module):
```python
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
```

- [ ] **Step 1: Write the failing tests**

`tests/test_reflection_attribution.py`:
```python
"""AttributedToolError: real per-tool attribution, replacing the "unknown_tool" guess
(PR5a/5b's carried-forward gap)."""

import asyncio

import pytest
from langchain_core.tools import StructuredTool

from pa_copilot.agents._react import AttributedToolError
from pa_copilot.reflection import attribute_tool_errors
from pa_copilot.rag.tool import search_clinical_guidance


def test_wraps_sync_tool_and_attributes_failure(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("index down")

    monkeypatch.setattr(search_clinical_guidance, "func", boom)
    (wrapped,) = attribute_tool_errors([search_clinical_guidance])
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
```

`tests/test_reflection_resilience.py`:
```python
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
```

> Adjust the `_settings(**overrides)` helper once you've confirmed `Settings` really is a
> plain frozen dataclass with a `__dict__` (it is, per `config.py` — `@dataclass(frozen=True)`
> instances still expose `__dict__`); the point of the helper is a fast settings object with
> a short timeout for the timeout test, not the exact construction mechanism.

`tests/test_config.py` — add assertions for the three new `Settings` fields and their
env-var overrides (`PA_WORKER_TIMEOUT_SECONDS`, `PA_MAX_TOOL_RETRIES`,
`PA_MODEL_AGENT_LITE`), mirroring however the existing `max_replans`/`model_agent` tests
in that file are already structured — read the file first.

- [ ] **Step 2: Run it red**

Run: `python -m pytest tests/test_reflection_attribution.py tests/test_reflection_resilience.py tests/test_config.py -v`
Expected: FAIL (`pa_copilot.reflection` doesn't exist yet; `Settings` has no new fields).

- [ ] **Step 3: Implement** the config changes, `_react.py` additions, and `reflection.py`
  exactly as specified above.

- [ ] **Step 4: Run it green + lint + full fast suite**

Run: `ruff check src tests && python -m pytest -q -m "not slow"`
Expected: new tests pass; **all 196 pre-existing tests still pass** (this task changes
`_react.py`'s except-chain ordering — verify nothing in `tests/test_react_helper.py` or any
worker test relies on the old ordering; if one does, that's a real finding for the task
reviewer, not something to silently work around).

- [ ] **Step 5: Commit**

```bash
git add config/routing.yaml config/models.yaml src/pa_copilot/config.py \
        src/pa_copilot/agents/_react.py src/pa_copilot/reflection.py \
        tests/test_config.py tests/test_reflection_attribution.py tests/test_reflection_resilience.py
git commit -m "feat(reflection): tool-error attribution + resilient worker-call wrapper"
```

---

### Task 2: Wire attribution + resilient calls into the four ReAct workers

**Files:**
- Modify: `src/pa_copilot/agents/intake.py`, `src/pa_copilot/agents/benefit_check.py`,
  `src/pa_copilot/agents/medical_necessity.py`, `src/pa_copilot/agents/decision_draft.py`
- Test: `tests/test_agent_intake.py`, `tests/test_agent_benefit_check.py`,
  `tests/test_agent_medical_necessity.py`, `tests/test_agent_decision_draft.py` (extend
  each — read each file in full first, since this task must not break any existing
  assertion in them)

**Interfaces:**

Each of the four worker `build_*_node` functions currently does:
```python
tools = [*(mcp_tools or []), <worker-specific extra tool, if any>]
...
_messages, result = await run_worker_react(
    model or get_agent_model(), tools, system_prompt=..., messages=[...],
    response_format=...,
)
```
This task changes each to:
```python
from pa_copilot.reflection import attribute_tool_errors, run_worker_react_resilient
...
tools = attribute_tool_errors([*(mcp_tools or []), <worker-specific extra tool, if any>])
...
_messages, result = await run_worker_react_resilient(
    model or get_agent_model(), tools, system_prompt=..., messages=[...],
    response_format=..., lite_model=get_lite_agent_model(),
)
```
`decision_draft.py`'s call (`tools=[]`) gets the same swap — `attribute_tool_errors([])`
is a no-op, and the lite-model fallback on `WorkerOutputError` is meaningful there too
(its structured-output call is the ONLY thing that can fail).

**No change to any worker's `except` clauses** — they already catch `WorkerToolError`,
`WorkerOutputError`, `WorkerRecursionError` by type; `run_worker_react_resilient` raises
exactly those same types (never a new one) once its own retry/fallback budget is spent.

- [ ] **Step 1: Verify langmem's memory tool shape** (the one library fact this plan's
  research session did not check — see Library facts). Before editing `intake.py`, run:

```python
python -c "
from pa_copilot.memory.store import memory_store
from pa_copilot.memory.tools import build_memory_tools
import tempfile
with memory_store(tempfile.mktemp(), embedder=None) as store:  # adjust ctor args to match
                                                                 # the real signature you find
                                                                 # in memory/store.py
    manage, search = build_memory_tools(store, ('pa','member','{member_id}'))
    print(type(search), 'func=', search.func, 'coroutine=', search.coroutine)
"
```
Record the actual result (func or coroutine, or neither) in this task's report. If it's
neither (some other `BaseTool` shape `attribute_tool_errors` can't reach), that's a real
gap for the task reviewer to see, not something to route around — the memory-search tool
would just keep falling back to `_best_effort_tool_name`'s guess, which is a defensible
`DONE_WITH_CONCERNS`, not a blocker (it's one tool among 2-4 per worker either way).

- [ ] **Step 2: Edit each of the four worker files** with the two-line change above (tool
  wrapping + call swap), importing `attribute_tool_errors`, `run_worker_react_resilient`
  from `pa_copilot.reflection` and `get_lite_agent_model` from `pa_copilot.agents._react`
  in each file that doesn't already import from `_react`.

- [ ] **Step 3: Extend each worker's existing test file** with one new test proving
  attribution actually reaches the worker's `tool_failures` update with the REAL tool
  name (not `"unknown_tool"`) — e.g. for `benefit_check.py` (only one tool, so today's
  `_best_effort_tool_name` already gets this right by luck; the real proof needs a worker
  with 2+ tools):

```python
# tests/test_agent_medical_necessity.py — new test
@pytest.mark.asyncio
async def test_tool_failure_is_attributed_to_the_real_tool_not_unknown(monkeypatch):
    """medical_necessity binds 2 tools (criteria_check + search_clinical_guidance) --
    exactly the case _best_effort_tool_name could never get right (PR5a/5b's carried
    gap). Force criteria_check to raise and assert the ToolFailure names it, not
    "unknown_tool"."""
    from langchain_core.tools import tool as lc_tool

    @lc_tool("criteria_check")
    def failing_criteria_check(service_code: str, diagnosis_codes: list) -> dict:
        """Fake criteria_check that always fails."""
        raise RuntimeError("mcp server down")

    fake = FakeToolCallingModel(script=[ai_tool_call("criteria_check", {
        "service_code": "72148", "diagnosis_codes": ["M54.16"],
    })], structured_responses=[])
    node = build_medical_necessity_node(mcp_tools=[failing_criteria_check], model=fake)
    update = await node({
        "request": PARequest(member_id="M1", service_code="72148", diagnosis_codes=["M54.16"],
                              requested_units=1, place_of_service="outpatient", provider_npi="1",
                              clinical_summary="x"),
        "benefit": BenefitResult(covered=True, plan_id="p", requires_pa=True, network_status="in_network"),
    })
    assert update["tool_failures"][0].tool == "criteria_check"
    assert update["tool_failures"][0].tool != "unknown_tool"
```
Adapt exact fixture construction to whatever this test file's existing tests already use
(read it first) — the point is the assertion (`.tool == "criteria_check"`, proven against
a worker bound to 2+ tools), not this exact scaffolding.

- [ ] **Step 4: Run it green + lint + full fast suite**

Run: `ruff check src tests && python -m pytest -q -m "not slow"`
Expected: all new + all 196 pre-existing tests pass. If `worker_timeout_seconds` (30s
default) makes any existing fast test slower than before (it shouldn't — `FakeToolCallingModel`
returns instantly, well under any timeout), investigate rather than raising the default.

- [ ] **Step 5: Commit**

```bash
git add src/pa_copilot/agents/intake.py src/pa_copilot/agents/benefit_check.py \
        src/pa_copilot/agents/medical_necessity.py src/pa_copilot/agents/decision_draft.py \
        tests/test_agent_intake.py tests/test_agent_benefit_check.py \
        tests/test_agent_medical_necessity.py tests/test_agent_decision_draft.py
git commit -m "feat(reflection): wire tool-error attribution + resilient calls into all four workers"
```

---

### Task 3: The replan/reroute state machine (`supervisor.py`) + `medical_necessity.py`'s low-confidence signal

**Files:**
- Modify: `src/pa_copilot/supervisor.py`, `src/pa_copilot/agents/medical_necessity.py`,
  `config/routing.yaml` (comment only — already done in Task 1, verify it stuck)
- Test: `tests/test_supervisor.py` (extend), `tests/test_agent_medical_necessity.py`
  (extend)

**Interfaces:**

`supervisor.py::hard_route` — broaden the cap-exhaustion condition (currently line ~54):
```python
def hard_route(state: PACaseState, *, settings: Settings | None = None) -> RouteTarget | None:
    s = settings or get_settings()

    if state.get("supervisor_hops", 0) >= s.max_hops:
        return "human_review"
    if (state.get("tool_failures") or state.get("needs_replan")) and state.get(
        "replan_count", 0
    ) >= s.max_replans:
        return "human_review"
    if state.get("decision") is not None and not state.get("needs_replan"):
        return "FINISH"
    if state.get("request") is None:
        return "intake"
    if state.get("benefit") is None:
        return "benefit_check"
    if state.get("necessity") is None:
        return "medical_necessity"
    return None
```
(Only the one `if` line changes — `tool_failures and ...` becomes
`(tool_failures or needs_replan) and ...`. Everything else in this function is unchanged.)

`supervisor.py::route_with_llm` — add a `settings` kwarg and a reflection hint:
```python
async def route_with_llm(
    state: PACaseState, *, model=None, settings: Settings | None = None
) -> RouterDecision:
    model = model or get_agent_model()
    s = settings or get_settings()
    summary = (state.get("context") or {}).get("running_summary")
    summary_text = getattr(summary, "summary", None) or "no summary yet"
    history = state.get("route_history") or []
    hint = ""
    if state.get("needs_replan"):
        necessity = state.get("necessity")
        if necessity is not None and (
            necessity.confidence < s.tau or necessity.criteria_status == "indeterminate"
        ):
            hint = (
                f"\nReflection hint: the last medical_necessity assessment was "
                f"low-confidence (confidence={necessity.confidence:.2f}, "
                f"status={necessity.criteria_status}). Consider routing back to "
                "medical_necessity to reassess with broader retrieval, or escalate to "
                "human_review if this has already been retried."
            )
        elif state.get("tool_failures"):
            hint = (
                "\nReflection hint: the last worker call failed a tool call. Routing "
                "back to the same phase will retry it; escalate to human_review if "
                "this has already been retried multiple times."
            )
    prompt = (
        "You are the routing supervisor for a prior-authorization case.\n"
        f"Case summary so far: {summary_text}\n"
        f"Still missing: {_missing_checklist(state)}\n"
        f"Recent routing history: {[h for h in history[-5:]]}"
        f"{hint}\n"
        "Choose the next node."
    )
    return await model.with_structured_output(RouterDecision).ainvoke([("user", prompt)])
```

`supervisor.py::build_supervisor_node` — consume `needs_replan` once per turn, before
routing, and clear it:
```python
def build_supervisor_node(*, model=None, settings: Settings | None = None) -> Callable:
    s = settings or get_settings()

    async def _node(state: PACaseState) -> dict:
        was_replanning = bool(state.get("needs_replan"))
        replan_count = state.get("replan_count", 0) + (1 if was_replanning else 0)
        effective_state = {**state, "replan_count": replan_count} if was_replanning else state

        target = hard_route(effective_state, settings=s)
        if target is not None:
            reason = f"deterministic guardrail -> {target}"
        else:
            decision = await route_with_llm(effective_state, model=model, settings=s)
            target = decision.next
            reason = decision.rationale

        step = RouteStep(
            from_node="supervisor", to_node=target, reason=reason,
            ts=datetime.now(timezone.utc).isoformat(),
        )
        update = {
            "next": target,
            "route_history": [step],
            "supervisor_hops": state.get("supervisor_hops", 0) + 1,
        }
        if was_replanning:
            update["replan_count"] = replan_count
            update["needs_replan"] = False
        return update

    return _node
```

`medical_necessity.py` — after a successful `run_worker_react_resilient` call, add the
low-confidence check (does **not** withhold `necessity` from state — see this plan's
Design decisions: the LLM router's fall-through already handles the loop-back choice;
withholding `necessity` would be a second, redundant, and *conflicting* mechanism):
```python
        try:
            _messages, necessity = await run_worker_react_resilient(
                model or get_agent_model(), tools, system_prompt=_SYSTEM_PROMPT,
                messages=[prompt], response_format=NecessityAssessment,
                lite_model=get_lite_agent_model(),
            )
        except WorkerToolError as exc:
            return tool_failure_update(exc)
        except (WorkerOutputError, WorkerRecursionError):
            return {"needs_replan": True}

        update = {"necessity": necessity, "retrieved_criteria": necessity.citations}
        settings = get_settings()
        if necessity.confidence < settings.tau or necessity.criteria_status == "indeterminate":
            update["needs_replan"] = True
        return update
```

- [ ] **Step 1: Write the failing tests**

`tests/test_supervisor.py` — add:
```python
def test_hard_route_exhausted_replans_goes_to_human_review_via_needs_replan_alone():
    """The broadened cap check (Task 3): a persistent LOW-CONFIDENCE loop (no
    ToolFailure at all) must still hit the cap -- not just the tool-failure path the
    pre-existing test above covers."""
    s = get_settings()
    state = {"request": {}, "needs_replan": True, "replan_count": s.max_replans}
    assert hard_route(state, settings=s) == "human_review"


@pytest.mark.asyncio
async def test_supervisor_node_increments_replan_count_and_clears_needs_replan():
    node = build_supervisor_node(model=None)
    update = await node({
        "request": {}, "benefit": {}, "necessity": {}, "needs_replan": True,
        "replan_count": 0, "supervisor_hops": 0,
    })
    assert update["replan_count"] == 1
    assert update["needs_replan"] is False


@pytest.mark.asyncio
async def test_supervisor_node_leaves_replan_count_untouched_when_not_replanning():
    node = build_supervisor_node(model=None)
    update = await node({"replan_count": 0})
    assert "replan_count" not in update
    assert "needs_replan" not in update
```

`tests/test_agent_medical_necessity.py` — add:
```python
@pytest.mark.asyncio
async def test_low_confidence_necessity_still_sets_necessity_but_flags_needs_replan():
    """Design ruling (this plan's Architecture section): low confidence does NOT
    withhold necessity from state -- the LLM router's existing fall-through already
    decides whether to loop back or escalate; withholding would be a second,
    conflicting mechanism. This must not regress tests/_full_case.py's ambiguous case."""
    necessity = NecessityAssessment(
        criteria_status="indeterminate", policy_id="PA-X", citations=[],
        unmet_requirements=[], confidence=0.2, rationale="unclear",
    )
    fake = FakeToolCallingModel(script=[
        ai_tool_call("criteria_check", {"service_code": "72148", "diagnosis_codes": ["M54.16"]}),
    ], structured_responses=[necessity])
    node = build_medical_necessity_node(mcp_tools=[FAKE_CRITERIA_CHECK_TOOL], model=fake)
    update = await node({"request": SOME_REQUEST, "benefit": SOME_BENEFIT})
    assert update["necessity"] == necessity
    assert update["needs_replan"] is True


@pytest.mark.asyncio
async def test_high_confidence_necessity_does_not_flag_needs_replan():
    necessity = NecessityAssessment(
        criteria_status="met", policy_id="PA-X", citations=[], unmet_requirements=[],
        confidence=0.9, rationale="clear",
    )
    ...  # same shape, assert "needs_replan" not in update
```
Adapt fixture construction (`SOME_REQUEST`, `SOME_BENEFIT`, the fake criteria_check tool)
to whatever this file's existing tests already build (read it first).

- [ ] **Step 2: Run it red**

Run: `python -m pytest tests/test_supervisor.py tests/test_agent_medical_necessity.py -v`

- [ ] **Step 3: Implement** the three edits above exactly as specified.

- [ ] **Step 4: Run it green + lint + FULL fast suite, paying specific attention to the
  AC-02/03/11 fixtures**

Run: `ruff check src tests && python -m pytest -q -m "not slow"`

Then, specifically:
```bash
python -m pytest -q tests/test_ac02_supervisor_routing.py tests/test_ac03_conditional_routing.py tests/test_ac11_agentic_rag.py -v
```
These three MUST still pass unchanged — they exercise `tests/_full_case.py`'s ambiguous
case, whose scripted `RouterDecision` picks `human_review` directly on the exact same
low-confidence `NecessityAssessment` this task now also flags `needs_replan=True` for. If
any of them fail, that is this task's own regression to fix here, not a finding to defer —
re-read this plan's Architecture section 3 and Design decisions before changing anything;
the intended behavior is that `needs_replan=True` is set AND counted/cleared by the very
next supervisor turn, but does **not** change which target that turn picks (the scripted
`RouterDecision` in the fixture is still consulted exactly as before, since `hard_route`
still returns `None` for that state — verify this is what actually happens, don't assume it).

- [ ] **Step 5: Commit**

```bash
git add src/pa_copilot/supervisor.py src/pa_copilot/agents/medical_necessity.py \
        tests/test_supervisor.py tests/test_agent_medical_necessity.py
git commit -m "feat(reflection): replan bookkeeping in supervisor + low-confidence needs_replan signal"
```

---

### Task 4: `human_review.py` resets `supervisor_hops`/`replan_count` on resume

**Files:**
- Modify: `src/pa_copilot/agents/human_review.py`
- Test: `tests/test_agent_human_review.py` (extend)

**Interfaces:**

```python
def build_human_review_node() -> Callable[[PACaseState], Awaitable[dict]]:
    async def _node(state: PACaseState) -> dict:
        payload = {
            "case_id": state.get("case_id"),
            "request": state["request"].model_dump() if state.get("request") else None,
            "necessity": state["necessity"].model_dump() if state.get("necessity") else None,
            "decision": state["decision"].model_dump() if state.get("decision") else None,
            "reason": "human review required",
        }
        resume_value = interrupt(payload)
        update = write_working_memory(state, "human_review_resume_value", resume_value)
        # A human just intervened -- treat the resumed leg of the case as a fresh
        # attempt budget (PR5b's carried-forward item, promoted to required PR6 work:
        # supervisor_hops never reset on resume, a live risk now that the recursion-limit
        # fix (PR5b) makes the MAX_HOPS cap actually reachable in production).
        update["supervisor_hops"] = 0
        update["replan_count"] = 0
        return update

    return _node
```

- [ ] **Step 1: Write the failing test** — extend
  `tests/test_human_review_interrupts_then_resumes` (or add a new test alongside it,
  following the existing file's exact throwaway-`StateGraph` + `InMemorySaver` pattern):

```python
@pytest.mark.asyncio
async def test_resume_resets_hop_and_replan_counters():
    node = build_human_review_node()
    g = StateGraph(dict)
    g.add_node("human_review", node)
    g.add_edge(START, "human_review")
    g.add_edge("human_review", END)
    compiled = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-resume-reset"}}

    await compiled.ainvoke(
        {"case_id": "case-0001", "supervisor_hops": 9, "replan_count": 2}, config=cfg
    )
    second = await compiled.ainvoke(Command(resume="approved-by-reviewer"), config=cfg)
    assert second["supervisor_hops"] == 0
    assert second["replan_count"] == 0
```

- [ ] **Step 2: Run it red**

Run: `python -m pytest tests/test_agent_human_review.py -v`

- [ ] **Step 3: Implement** the two added lines in `human_review.py` exactly as specified.

- [ ] **Step 4: Run it green + lint + full fast suite**

Run: `ruff check src tests && python -m pytest -q -m "not slow"`

- [ ] **Step 5: Commit**

```bash
git add src/pa_copilot/agents/human_review.py tests/test_agent_human_review.py
git commit -m "fix(reflection): reset supervisor_hops/replan_count on human_review resume"
```

---

### Task 5: AC-12 full in-graph evidence

**Files:**
- Create: `tests/_reflection_case.py`, `tests/test_ac12_reflection.py`,
  `scripts/reflection_demo.py`
- Create (committed evidence): `traces/reflection_tool_failure.json`,
  `traces/reflection_low_confidence.json`

**Read first:** `tests/_full_case.py` in full (already read this session — reuse its
`ScriptedCase` dataclass shape, `new_initial_state`, `build_stub_summarization_node`,
`fake_rag_embedder` helpers directly via import rather than duplicating them) and
`scripts/full_case_demo.py` in full (mirror its `sys.path` setup, `_dump_route_step`
stripping `ts`, `_write` helper with `encoding="utf-8", newline="\n"` — reuse these two
functions via import from `scripts/full_case_demo.py` rather than re-defining them, since
they're already exactly right and duplicating byte-stable-serialization logic is how this
project's evidence bugs happen — see PR5b's own final-review finding about timestamps
leaking into committed traces).

**Interfaces:**

`tests/_reflection_case.py` — two new scripted cases, same member/service/provider as
`_full_case.py` for consistency (not required to match, but avoids inventing new synthetic
IDs with no corpus backing):

1. **Tool-failure case**: a fake `benefit_lookup` tool that raises on its first call and
   succeeds on its second (a closure over a mutable counter, the simplest deterministic way
   to simulate a transient failure that recovers on retry — do NOT rely on
   `run_worker_react_resilient`'s own tenacity retries to mask this at the tool level for
   THIS evidence; the point of this trace is the supervisor-level reroute, i.e. the failure
   must survive past `max_tool_retries` tenacity attempts and reach `tool_failure_update`,
   so the fake tool needs to fail more times than `max_tool_retries` on its first "visit"
   from the graph and then succeed once the supervisor reroutes back). Concretely: fail
   unconditionally on attempts 1 through `max_tool_retries`, succeed from attempt
   `max_tool_retries + 1` onward — reached only via the supervisor's reroute-to-same-worker,
   not via tenacity's own in-turn retries.
2. **Low-confidence case**: `medical_necessity`'s scripted `NecessityAssessment` is
   low-confidence on its first visit, then the supervisor's scripted `RouterDecision`
   explicitly routes back to `medical_necessity` (proving the loop-back this plan's
   Architecture section 3 says the LLM router — not a hard guardrail — is responsible
   for), and `medical_necessity`'s SECOND scripted `NecessityAssessment` is high-confidence,
   reaching `decision_draft`/`FINISH` normally.

Each case also needs an **exhaustion** variant reused by the test file (persistent failure
past `max_replans` -> `human_review`) — write these as functions returning a
`ScriptedCase`-like object parameterized by `recovers: bool`, rather than four near-duplicate
functions.

`tests/test_ac12_reflection.py` — drives all four scenarios (tool-failure recovers,
tool-failure exhausts, low-confidence recovers, low-confidence exhausts) through the REAL
compiled graph (`pa_copilot.graph.make_graph`), asserting:
- Recovery cases: final `route_history` shows >=2 visits to the failing/looping worker,
  `decision` is set, `next == "FINISH"`, `tool_failures`/`replan_count` show exactly the
  expected number of recorded attempts.
- Exhaustion cases: final route reaches `human_review` (`"__interrupt__" in result`),
  `replan_count == settings.max_replans`, and — the NFR-07-relevant assertion — the whole
  `graph.ainvoke(...)` call completes without raising; the ONLY way a persistently failing
  tool can end a run is a clean route to `human_review`, never an unhandled exception
  escaping `ainvoke`.

`scripts/reflection_demo.py` — mirrors `scripts/full_case_demo.py` structure exactly
(same `sys.path` setup, same `_open_memory_store` pattern via import, same `_write`
helper) and writes:
- `traces/reflection_tool_failure.json`: the tool-failure-recovers case's full
  `route_history` (via `full_case_demo._dump_route_step`, stripped of `ts`) plus the final
  `tool_failures` list (each `.model_dump()`, also stripped of `ts` the same way) and
  `replan_count`.
- `traces/reflection_low_confidence.json`: the low-confidence-recovers case's full
  `route_history`, both `NecessityAssessment.model_dump()`s (first low-confidence, second
  high-confidence), and `replan_count`.

Both files use `schema_version` from `full_case_demo.TRACE_SCHEMA_VERSION` for
consistency with the existing `traces/run_full_case.json` etc.

- [ ] **Step 1: Write `tests/_reflection_case.py`** with the scripted cases described
  above, importing shared scaffolding from `tests/_full_case.py` rather than duplicating
  it.

- [ ] **Step 2: Write `tests/test_ac12_reflection.py`** with the four scenario tests. Run
  it red first (`python -m pytest tests/test_ac12_reflection.py -v`) — expect failures
  either from missing scaffolding or from surfacing a real gap in Tasks 1-4's
  implementation; if a scenario reveals a genuine bug in the reflection mechanism itself
  (not a test-scaffolding mistake), that's exactly what this task exists to catch — fix
  the root cause in the relevant module (Task 1-4's files), not by weakening this test.

- [ ] **Step 3: Get all four scenarios green.**

Run: `python -m pytest tests/test_ac12_reflection.py -v`

- [ ] **Step 4: Write and run `scripts/reflection_demo.py`**

Run: `python scripts/reflection_demo.py`
Confirm both trace files are written, then run it a SECOND time and `git diff --stat
traces/` to confirm byte-stability (empty diff) before committing.

- [ ] **Step 5: Full fast suite + lint**

Run: `ruff check src tests scripts && python -m pytest -q -m "not slow"`

- [ ] **Step 6: Commit**

```bash
git add tests/_reflection_case.py tests/test_ac12_reflection.py scripts/reflection_demo.py \
        traces/reflection_tool_failure.json traces/reflection_low_confidence.json
git commit -m "test(reflection): AC-12 full in-graph evidence (tool-failure + low-confidence, recover + exhaust)"
```

---

### Task 6: NFR-07 degradation test + ledger/spec updates + whole-branch pass

**Files:**
- Create: `tests/test_nfr07_degradation.py`
- Modify: `specs/acceptance-criteria.md`, `specs/nfr.md`

**Interfaces:**

`tests/test_nfr07_degradation.py` — NFR-07's own wording ("a persistently failing tool
ends at `human_review`, never an unhandled exception") is already proven by Task 5's
exhaustion scenarios; this file's job is to be NFR-07's own named evidence file per
`specs/nfr.md`'s table (test id traceability, `tests/test_ac_traceability.py` checks this),
not to re-derive new scenarios. Import and re-run (or directly reference) the tool-failure
exhaustion case from `tests/_reflection_case.py`, with an NFR-07-flavored docstring and
assertion focus (no unhandled exception; `asyncio.wait_for` timeout path specifically —
add one dedicated case here that a `benefit_lookup` fake tool that just hangs
(`await asyncio.sleep(999)`, monkeypatched into a real async tool) still ends the run at
`human_review` within a bounded wall-clock time, proving the timeout path specifically
rather than only the exception path Task 5 already covers).

- [ ] **Step 1: Write the failing test** (the hang/timeout scenario described above; reuse
  `tests/_reflection_case.py`'s tool-failure exhaustion scenario for the "ends at
  human_review" half via a direct import/call, don't duplicate its construction).

- [ ] **Step 2: Run it red, then green.**

Run: `python -m pytest tests/test_nfr07_degradation.py -v`

- [ ] **Step 3: Update `specs/acceptance-criteria.md`** — AC-12 row → `done`, with a short
  PR6 evidence note (both test files, both trace files, naming the recover+exhaust split
  explicitly since that's the actual coverage, not just "a test exists").

- [ ] **Step 4: Update `specs/nfr.md`** — NFR-07 row → `done`, evidence note naming
  `tests/test_nfr07_degradation.py` + the timeout-specific case, and cross-referencing
  Task 5's exhaustion scenarios rather than re-describing them.

- [ ] **Step 5: Full whole-branch pass**

Run: `ruff check src tests scripts && python -m pytest -q -m "not slow"`
Run: `python -m pytest -q` (include slow, confirm nothing unexpectedly marked slow)

- [ ] **Step 6: Commit**

```bash
git add tests/test_nfr07_degradation.py specs/acceptance-criteria.md specs/nfr.md
git commit -m "test(reflection): NFR-07 degradation evidence + AC-12/NFR-07 ledger updates"
```

---

## After this branch: handoff to PR7

Update `docs/BUILD_LOG.md` (mirror PR1-5b's entries) once merged. PR7 (`feat/interfaces`)
scope per design.md §10: `cli.py` (`pac` — `ingest`, `submit`, `resume`, `memory`,
`persistence-test`, `compare`, `demo`, `all`), `app/streamlit_app.py`, the full genuine
MCP transcript (needs a real `GEMINI_API_KEY`, currently representative), `single_agent.py`
+ `pac compare`, `docs/{single-vs-multi-agent,agent-patterns,rubric-coverage}.md`, README
quick-start, CI (`.github/workflows/tests.yml`). Closes AC-10 (full), NFR-01, NFR-02,
NFR-04 (full trace-schema validation across `traces/`), NFR-06.

**PR7 must also implement, now that PR6 provides the mechanism it needs:** `pac resume`
should rely on `human_review.py`'s now-existing `supervisor_hops`/`replan_count` reset
(Task 4) rather than re-inventing hop-budget handling in the CLI layer — the CLI's job is
just to call `graph.ainvoke(Command(resume=...), config=...)` on a fresh process, per
`docs/design.md` §3.7's existing `pac resume` contract; no new state-reset logic belongs
in `cli.py` itself.

**Not carried forward as an open gap** (this PR is the one that closes it): PR5a/PR5b's
`_best_effort_tool_name`'s universal `"unknown_tool"` fallback for any 2+-tool worker —
closed by Task 1/2's `AttributedToolError` mechanism, verified against the MCP-adapter
tool shape, the RAG tool shape, and (Task 2, Step 1) the langmem memory-tool shape.

**Still open, not assigned to a PR** (unchanged from PR5b's own handoff, PR6 doesn't touch
either): `state["messages"]` is still never written by any worker (`SummarizationNode`
runs on an empty thread in the real system); `decision_draft` only writes memory on
`disposition == "deny"` (approve/refer paths leave no long-term memory trace).
