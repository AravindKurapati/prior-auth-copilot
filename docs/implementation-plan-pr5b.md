# PR5b — Graph Core, Part B: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The second half of PR5. Builds the remaining three workers
(`medical_necessity`, `decision_draft`, `human_review`), the actual hand-rolled `graph.py`
`StateGraph` topology, and the checkpointer + store wiring. Closes **AC-01, AC-02, AC-03,
AC-04, AC-05** in full, brings **AC-10, AC-11** to their full in-graph proof, and closes
**NFR-03**'s full "disposition not forced" behavioral proof. Branch `feat/graph-core-b`,
off `main` @ the PR5a merge commit.

**Architecture:** `medical_necessity` and `decision_draft` follow PR5a's established
worker shape exactly (`build_X_node(...) -> node callable`, built on
`agents/_react.py::run_worker_react`) — no new worker infrastructure needed.
`medical_necessity` binds **both** the MCP `criteria_check` tool and PR3's
`search_clinical_guidance` `@tool` into one tool list; retrieval stays "agentic, not a
fixed step" (design.md §7 / AC-11) by teaching the MCP-status → action mapping
(design.md §4.1) in the system prompt and letting the model itself decide whether to call
the RAG tool, rather than our code pre-gating tool availability — `should_search_guidance`
(PR3) is already tested at the predicate level and is referenced in the prompt, not
re-implemented as a hard gate here. `decision_draft` needs no external tools (`tools=[]`
through `run_worker_react` — verified in PR5a that `create_react_agent` handles an empty
tool list) and does its self-critique (checking every `cited_criteria` quote is present in
`state["retrieved_criteria"]`) as plain Python after the structured-output call returns;
the *enforcement* of a needs_replan → reroute-to-medical_necessity loop (with a retry cap)
is PR6's `reflection.py`, not this PR — `decision_draft` only detects the mismatch and
raises the flag. `human_review` is a two-line `interrupt()` stub. `graph.py` assembles all
five workers + `summarize` + `supervisor` into one `StateGraph`, compiled with
`AsyncSqliteSaver` (**not** `SqliteSaver` — see Library facts below, this is the one
correction to design.md's stack table this PR makes) and the PR4/PR5a `PolicyStore`. Two
new evidence scripts (mirroring PR4's `scripts/run_persistence_test.py` pattern) produce
`traces/pause_resume_transcript.md` (two real separate `python` process invocations) and
the AC-02/03/10/11 full-graph-run traces (deterministic, `FakeToolCallingModel`-driven,
since no `GEMINI_API_KEY` exists in this environment — a `@pytest.mark.slow +
skipif(not GEMINI_API_KEY)` counterpart is added for the user to regenerate genuinely later,
matching PR2/PR3's precedent for MCP/RAG evidence).

**Tech Stack additions over PR5a:** `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`
(not the sync `SqliteSaver` design.md's stack table names — verified incompatible with
this project's fully-async graph, see Library facts), `aiosqlite` (already an installed
transitive dependency via `langgraph-checkpoint-sqlite`, pinned in `constraints.txt`, but
not yet declared in `pyproject.toml`'s own `dependencies` — this PR adds it explicitly),
`grandalf` (dev-only, for `Graph.draw_ascii()` — design.md's literal evidence format for
`traces/graph_topology.txt`; verified `draw_ascii()` raises `ImportError` without it, and
`draw_mermaid()` works without any extra dependency but is not what design.md specifies).

**Spec:** `docs/design.md` §3 (the graph, authoritative), §4.1 (MCP status mapping
table), §7 (agentic RAG), §6.1-6.3 (memory namespaces/writes — for `decision_draft`'s
critical write). `specs/acceptance-criteria.md` AC-01/02/03/04/05/10/11,
`specs/nfr.md` NFR-03. Design decisions this session (2026-09-11, continued from PR5a):
`decision_draft`'s critical-importance write is a direct `PolicyStore.put(...,
value={"importance": "critical", ...})` call into `("pa", "member", <member_id>)`, not a
new tool (matches the ruling recorded in PR5a's `docs/BUILD_LOG.md` handoff).

## Global Constraints

- Python `>=3.11`. Package `pa_copilot` under `src/`. Branch `feat/graph-core-b` off
  `main` @ the PR5a merge commit (confirm with `git log --first-parent` before starting).
- Stack pins unchanged except the two additions above (`aiosqlite`, `grandalf`) — do not
  otherwise touch `langgraph>=1.0,<2` / `langchain-core>=0.3,<0.4` pins.
- **No `GEMINI_API_KEY` / `GOOGLE_API_KEY` in this environment.** Every fast test is
  deterministic against `tests/_fakes.py::FakeToolCallingModel` (PR5a). Full-graph-run
  traces get a deterministic producer script (committed, byte-stable) plus a
  `@pytest.mark.slow + skipif(not GEMINI_API_KEY)` test that the user regenerates with a
  real key later, per this project's established pattern (PR2's `mcp_toolcall_transcript.md`,
  PR3's `agentic_rag_decision.md`).
- Test output must be **pristine** (`pyproject.toml` `filterwarnings = ["error"]` plus
  existing narrow ignores — 4 entries as of PR5a's merge). If `AsyncSqliteSaver` or
  `aiosqlite` emit any import/runtime warning, add one more narrow, category+module-scoped
  ignore — do not broaden an existing one (PR5a Task 1's fix-round lesson: a bare
  category-only ignore silently shadows an unrelated entry for the same category).
- Committed evidence under `traces/` stays byte-stable (no timestamps/randomness in
  committed bytes; `write_text(..., encoding="utf-8", newline="\n")`).
- Windows: `AsyncSqliteSaver`/`PolicyStore` both own live connections. Never
  `shutil.rmtree` a db file whose connection is open — close explicitly (mirrors PR4's
  `memory_store` fixture pattern, and the persistence-test script's process-boundary
  closes).
- Work on branch `feat/graph-core-b`. The controller does the `--no-ff` merge after the
  whole-branch review — do NOT merge in a task.
- TDD: failing test first, run red, minimal impl, run green, commit. Frequent commits.

## Interfaces from PR1–PR5a (on `main` after the PR5a merge)

- `pa_copilot.config.get_settings() -> Settings` — `.state_db` (`"./.pa_state.db"`),
  `.memory_db` (`"./.pa_memory.db"`), `.model_agent`, `.max_hops`, `.max_replans`, `.tau`
  (`0.55`), `.rag_min_score`.
- `pa_copilot.state.PACaseState` — all fields from PR1-5a: `messages`, `case_id`,
  `session_id`, `member_id`, `raw_provider_text`, `quarantine_ref`, `request`, `benefit`,
  `necessity`, `decision`, `retrieved_criteria: list[CriteriaCitation]`, `next`,
  `route_history`, `confidence`, `needs_replan`, `replan_count`, `supervisor_hops`,
  `tool_failures`, `working_memory`, `context`, `summarized_messages`.
  `new_case_state(case_id, session_id, member_id, raw_provider_text) -> PACaseState`.
- `pa_copilot.schemas` — `NecessityAssessment` (`criteria_status:
  Literal["met","not_met","indeterminate"]`, `policy_id: str|None`, `citations:
  list[CriteriaCitation]`, `unmet_requirements: list[str]`, `confidence: float`,
  `rationale: str`); `PADecision` (`disposition:
  Literal["approve","deny","refer_clinical_review"]`, `cited_criteria:
  list[CriteriaCitation]`, `reviewer_summary: str`, `confidence: float`,
  `human_review_required: bool = True`); `CriteriaCitation` (`source:
  Literal["mcp_resource","rag_corpus"]`, `clause_id`, `quote`, `relevance`).
- `pa_copilot.agents._react` — `get_agent_model(...)`, `WorkerToolError`,
  `WorkerOutputError`, `WorkerRecursionError` (all with `.error`; `WorkerToolError` also
  `.tool`, `.attempt`), `run_worker_react(model, tools, *, system_prompt, messages,
  response_format, config=None, recursion_limit=10) -> (messages, structured)`,
  `tool_failure_update(exc: WorkerToolError) -> dict`.
- `pa_copilot.supervisor.hard_route(state, *, settings=None) -> RouteTarget | None` and
  `build_supervisor_node(*, model=None, settings=None) -> node callable`.
- `pa_copilot.agents.intake.build_intake_node(*, store, mcp_tools=None, model=None)`,
  `pa_copilot.agents.benefit_check.build_benefit_check_node(*, mcp_tools=None, model=None)`
  — both return `async def _node(state) -> dict`.
- `pa_copilot.context.assembly.select_for(node, state) -> dict` (`_SELECTORS` covers
  `"intake"`, `"benefit_check"`, `"medical_necessity"` — already returns
  `{"request", "benefit", "retrieved_criteria"}` with `retrieved_criteria` defaulted to
  `[]` — and `"decision_draft"` — `{"request", "benefit", "necessity"}`, never
  `raw_provider_text`). `write_working_memory(state, key, value) -> dict`.
- `pa_copilot.context.summarization.summarize(state) -> dict` — the graph-node-shaped
  wrapper this PR wires directly into `graph.py`.
- `pa_copilot.mcp_client` — `load_pa_tools(client=None, session=None) -> list[BaseTool]`
  (loads `benefit_lookup`, `provider_lookup`, `criteria_check`); `pa_session(client=None)`
  (async contextmanager, one long-lived stdio session); `build_client(env=None, cwd=None)`.
- `pa_copilot.rag.tool.search_clinical_guidance` (a `@tool`), `should_search_guidance
  (criteria_status, *, unmet_requirements=None) -> bool` (pure predicate, PR3).
- `pa_copilot.memory.store.PolicyStore` (now with a real `abatch`, PR5a fix),
  `open_memory_store(...)`, `memory_store(...)` (contextmanager). `PolicyStore.put(namespace,
  key, value, *, ttl=NOT_PROVIDED)` — pass `value={"importance": "critical", ...}` to
  bypass the tool's routine-only default.
- `pa_copilot.tracing.RunTracer(trace_dir, case_id, redact_values, session_id=None)`,
  `.event(node, kind, payload)`, `.finish(decision=None) -> Path`.
- `tests/_fakes.py::FakeToolCallingModel(script=[...], structured_responses=[...])`,
  `ai_tool_call(name, args, *, call_id=None)` (PR5a's fix-wave rename — no longer `id`).
- `tests/conftest.py` fixtures: `memory_store` (real `PolicyStore` over `tmp_path` +
  `FakeEmbedder`), `sample_request`, `tmp_trace_dir`, `frozen_now`.

## Library facts verified for this plan (2026-09-11, installed versions, empirically
checked in this repo's venv)

- Installed: `langgraph==1.0.1`, `langgraph-checkpoint-sqlite==3.0.3`,
  `aiosqlite==0.22.1` (already installed + pinned in `constraints.txt`, but **not** yet a
  direct `pyproject.toml` dependency — add it explicitly, since this PR is the first to
  import it directly rather than relying on it being pulled in transitively).
- **`SqliteSaver` (`langgraph.checkpoint.sqlite.SqliteSaver`, the sync checkpointer
  design.md's §1 stack table names) does not support ANY async operation.** Confirmed by
  running it: `graph.ainvoke(...)` against a graph compiled with a plain `SqliteSaver`
  raises `NotImplementedError: The SqliteSaver does not support async methods. Consider
  using AsyncSqliteSaver instead.` at the very first checkpoint read
  (`AsyncPregelLoop.__aenter__` → `checkpointer.aget_tuple(...)`). Since `graph.py`'s
  `make_graph()` is async and every worker is an async ReAct loop (design.md itself:
  "Graph build is async... CLI wraps it in `asyncio.run`"), the graph must be invoked via
  `.ainvoke()`/`.astream()`, which makes the sync `SqliteSaver` **completely unusable**
  here — this is not a corner case, it fails on the very first turn. Use
  `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver` instead.
- `AsyncSqliteSaver.__init__(self, conn: aiosqlite.Connection, *, serde=None)`;
  `AsyncSqliteSaver.from_conn_string(conn_string) -> AsyncIterator[AsyncSqliteSaver]` is an
  **async contextmanager that closes the connection on exit** (same "unusable for a
  long-lived graph" gotcha PR4 found for `SqliteStore.from_conn_string` — construct
  directly instead). `await saver.setup()` is itself a coroutine (confirmed via
  `inspect.iscoroutinefunction`).
- **Verified end-to-end** (script run this session, two separate `asyncio.run()` calls
  simulating two separate process invocations against one on-disk sqlite file): `conn =
  await aiosqlite.connect(path)`, `saver = AsyncSqliteSaver(conn)`, `await saver.setup()`,
  compile with `checkpointer=saver`. First invocation: `await graph.ainvoke(initial_state,
  config={"configurable": {"thread_id": t}})` runs to the `interrupt()` call and returns
  a result dict containing `"__interrupt__": [Interrupt(value=..., id=...)]` — **the
  interrupted state is a normal return value, not a raised exception the caller must
  catch.** `await graph.aget_state(config)` returns an object with `.next` — a tuple
  naming the paused node(s) (confirmed: `('paused',)`). Second invocation, on a **fresh**
  connection/graph object built the same way against the same file: `await
  graph.ainvoke(Command(resume=<value>), config={"configurable": {"thread_id": t}})`
  (same `thread_id`) resumes correctly and completes. `Command(resume=...)` and
  `interrupt(value)` both live in `langgraph.types`.
- `StateGraph.add_conditional_edges(source, path_fn, path_map: dict[Hashable, str])` —
  `path_map` maps whatever `path_fn(state)` returns to a real node name; `END` (from
  `langgraph.graph`) is a valid target for `"FINISH"`.
- `StateGraph.compile(checkpointer=None, *, cache=None, store=None, interrupt_before=None,
  interrupt_after=None, debug=False, name=None) -> CompiledStateGraph` — `store=` accepts
  any `BaseStore`, including the sync `PolicyStore` (its `abatch` override, added in
  PR5a's final-review fix wave, is exactly what makes it usable from an async graph —
  this PR is the first to actually exercise that in a full `.ainvoke()` graph run, not
  just an isolated tool call).
- `CompiledStateGraph.get_graph() -> langchain_core.runnables.graph.Graph` —
  `.draw_ascii()` (design.md's literal spec for `traces/graph_topology.txt`) raises
  `ImportError: Install grandalf to draw graphs` without the `grandalf` package installed
  (confirmed — not installed in this venv). `.draw_mermaid()` works with no extra
  dependency and needs no image renderer, but is not what design.md specifies. This plan
  adds `grandalf` as a dev dependency rather than silently switching formats.
- `rag/tool.py::search_clinical_guidance`'s `except RagIndexUnavailable:` (line 170) does
  **not** catch `CorporaUnavailable`, which `_service_name()` (called at line 157, inside
  the same `try` block, on the corrective-rewrite path) can raise via
  `data_access.list_policies() -> load_corpora()`. This is exactly the item PR3's
  `docs/BUILD_LOG.md` parked: *"medical_necessity's tool-call try/except should widen to
  `(RagIndexUnavailable, CorporaUnavailable)` → `[]`."* Verified the fix belongs in
  `rag/tool.py` itself (widen the except tuple there), not in `medical_necessity` — this
  keeps the tool's existing "degrades to `[]`, never raises to the agent loop" contract
  uniform, so no caller needs RAG-specific exception handling.

## File Structure

| File | Responsibility |
|---|---|
| `src/pa_copilot/rag/tool.py` | widen `except RagIndexUnavailable` → `except (RagIndexUnavailable, CorporaUnavailable)` (PR3 park, closed) |
| `pyproject.toml` | add `aiosqlite>=0.20` to `dependencies`; add `grandalf` to `dev` optional-dependencies |
| `src/pa_copilot/agents/medical_necessity.py` | `build_medical_necessity_node(*, mcp_tools=None, model=None) -> node callable`; binds `criteria_check` + `search_clinical_guidance`, emits `NecessityAssessment` |
| `src/pa_copilot/agents/decision_draft.py` | `build_decision_draft_node(*, store, model=None) -> node callable`; synthesizes `PADecision`, self-critiques citations, writes critical memory on `deny` |
| `src/pa_copilot/agents/human_review.py` | `build_human_review_node() -> node callable`; calls `interrupt(...)` |
| `src/pa_copilot/graph.py` | `make_graph(*, store, checkpointer, mcp_tools, model=None) -> CompiledStateGraph`; the hand-rolled topology |
| `scripts/graph_topology_demo.py` | writes `traces/graph_topology.txt` (`get_graph().draw_ascii()`) from a graph built with fakes — no real checkpointer/store file needed |
| `scripts/run_pause_resume_test.py` | AC-05 evidence: spawns two separate `python` processes sharing one `.pa_state.db` / `.pa_memory.db`, combined stdout → `traces/pause_resume_transcript.md` |
| `scripts/full_case_demo.py` | AC-02/03/10/11 evidence producer: runs a clear-cut case and an ambiguous case through the real compiled `graph.py`, `FakeToolCallingModel`-driven → `traces/run_full_case.json`, `traces/route_clearcut.json`, `traces/route_ambiguous.json` |
| `tests/test_rag_tool_corpora_unavailable.py` | regression test for the widened except clause |
| `tests/test_agent_medical_necessity.py` | happy path (met/not_met/indeterminate), RAG-called vs. not-called, tool-failure path |
| `tests/test_agent_decision_draft.py` | happy path, citation-mismatch → `needs_replan`, critical memory write on `deny` |
| `tests/test_agent_human_review.py` | interrupt pauses; resume value flows into the node's return |
| `tests/test_graph_topology.py` | node/edge shape assertions on the compiled graph (AC-01) |
| `tests/test_ac01_typed_state.py` | `PACaseState` is a `TypedDict`; every node reads/writes it; `graph.get_graph()` introspection |
| `tests/test_ac02_supervisor_routing.py` | supervisor emits `RouterDecision`; all four (well, five) workers reachable; a run visits ≥2 workers |
| `tests/test_ac03_conditional_routing.py` | clear-cut case → auto path to `decision_draft`; ambiguous case → `human_review` |
| `tests/test_ac04_structured_output.py` | each worker returns a validated model; malformed output is handled |
| `tests/test_ac05_checkpointer.py` | interrupt at `human_review`; a **fresh graph object** built from the same `AsyncSqliteSaver` file resumes and completes |
| `tests/test_ac10_mcp_integration.py` | a graph run where `medical_necessity` invokes `criteria_check` through the adapter |
| `tests/test_ac11_agentic_rag.py` (extends PR3's) | full in-graph "agent decides" run: one case calls RAG, one doesn't |
| `tests/test_nfr03_quarantine.py` | full closure: injection canary → disposition not forced, `quarantine_ref` recorded |
| `specs/acceptance-criteria.md`, `specs/nfr.md` | AC-01/02/03/04/05/10/11 → `done`; NFR-03 → `done` |
| `docs/design.md` | §1 stack table: `SqliteSaver` → `AsyncSqliteSaver` (correction) |

---

### Task 1: Small fixes carried from PR3/PR5a park notes

**Files:**
- Modify: `src/pa_copilot/rag/tool.py`
- Modify: `pyproject.toml`
- Test: `tests/test_rag_tool_corpora_unavailable.py` (create)

**Interfaces:**
- `search_clinical_guidance`'s `except (RagIndexUnavailable, CorporaUnavailable):` now
  catches both; behavior unchanged otherwise (`return []`, same warning log — extend the
  log message to name whichever exception fired, or keep it generic, your call).
- `pyproject.toml` `dependencies`: add `"aiosqlite>=0.20"`. `[project.optional-dependencies]
  dev`: add `"grandalf>=0.8"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag_tool_corpora_unavailable.py
"""PR3 park, closed in PR5b: search_clinical_guidance's corrective-rewrite path
calls _service_name() -> list_policies() -> load_corpora(), which can raise
CorporaUnavailable — a different exception than the tool's own
RagIndexUnavailable guard. Both must degrade to [] uniformly."""

import pytest

from pa_copilot.mcp_server.data_access import CorporaUnavailable
from pa_copilot.rag import tool as T


def test_corpora_unavailable_during_rewrite_degrades_to_empty(monkeypatch, fake_embedder):
    T.set_tool_embedder(fake_embedder)

    def boom(hits, **kw):
        return []  # force the corrective-rewrite path

    monkeypatch.setattr(T.index, "search", lambda *a, **kw: [])

    def raise_corpora_unavailable(service_code):
        raise CorporaUnavailable("synthetic corpora missing")

    monkeypatch.setattr(T, "_service_name", raise_corpora_unavailable)

    result = T.search_clinical_guidance.invoke({"query": "anything", "service_code": "72148"})
    assert result == []
    T.reset_tool_embedder()
```

> Adjust the exact monkeypatch shape once you see how `index.search` is actually called
> (it's called twice — original then rewrite — inside the tool); the point of this test is
> that a `CorporaUnavailable` raised from `_service_name` during the rewrite path is
> caught and degrades to `[]`, not that every mock call matches this sketch verbatim.

- [ ] **Step 2: Run it red**

Run: `python -m pytest tests/test_rag_tool_corpora_unavailable.py -v`
Expected: FAIL (`CorporaUnavailable` propagates, uncaught).

- [ ] **Step 3: Widen the except clause**

`src/pa_copilot/rag/tool.py` — add the import and widen line 170:

```python
from pa_copilot.mcp_server.data_access import CorporaUnavailable
# ...
    except (RagIndexUnavailable, CorporaUnavailable):
```

- [ ] **Step 4: Add the two new dependencies**

`pyproject.toml`:

```toml
dependencies = [
    ...
    "tenacity>=8.3",
    "aiosqlite>=0.20",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.24,<2", "pytest-mock>=3.14", "ruff>=0.6", "grandalf>=0.8"]
```

Run `pip install -e ".[dev]" -c constraints.txt` (or equivalent) to confirm `grandalf`
installs cleanly; add its resolved version to `constraints.txt` alongside the existing
`aiosqlite==0.22.1` entry (already there) — no change needed for `aiosqlite` itself since
it's already pinned.

- [ ] **Step 5: Run it green + lint + fast suite + commit**

Run: `ruff check src tests && python -m pytest -q -m "not slow"`

```bash
git add src/pa_copilot/rag/tool.py pyproject.toml constraints.txt \
        tests/test_rag_tool_corpora_unavailable.py
git commit -m "fix(rag): widen search_clinical_guidance's except to CorporaUnavailable (PR3 park)"
```

---

### Task 2: `agents/medical_necessity.py`

**Files:**
- Create: `src/pa_copilot/agents/medical_necessity.py`
- Test: `tests/test_agent_medical_necessity.py` (create)

**Interfaces:**
- `build_medical_necessity_node(*, mcp_tools: list | None = None, model=None) -> node
  callable` — same shape as `benefit_check`. `mcp_tools` is expected to contain (at least)
  `criteria_check`; the RAG tool `search_clinical_guidance` is imported directly (it's
  already a `@tool`, not loaded via MCP) and always appended to the bound tool list.
  `view = select_for("medical_necessity", state)` (already returns `request`, `benefit`,
  `retrieved_criteria` defaulted to `[]` — PR5a's Task 4 fix). Build a `HumanMessage`
  summarizing the request + benefit result. System prompt teaches the design.md §4.1
  MCP-status → clinical-judgment mapping table verbatim and instructs the model: call
  `search_clinical_guidance` when `criteria_check`'s status is `not_found`/`indeterminate`,
  or when a diagnosis-based `excluded` result still needs narrative confirmation, or when
  unmet requirements need interpretation — i.e., teach `should_search_guidance`'s logic in
  prose rather than gating tool availability in code (AC-11: "the agent decides").
  `response_format=NecessityAssessment`. On success, return `{"necessity": <assessment>,
  "retrieved_criteria": <assessment.citations>}` (so `decision_draft`'s later
  `select_for` view sees the citations under the dedicated state field, matching
  design.md's `retrieved_criteria: list[CriteriaCitation]` top-level field, not nested
  only inside `necessity`). On `WorkerToolError`/`WorkerOutputError`/`WorkerRecursionError`,
  use `tool_failure_update(...)` (only defined for `WorkerToolError` in PR5a — for the
  other two, return a similarly-shaped `{"needs_replan": True}` update without a
  `ToolFailure` record, since those aren't tool failures; check `agents/_react.py`'s
  actual exception surface from PR5a before finalizing this branch).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_medical_necessity.py
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from _fakes import FakeToolCallingModel, ai_tool_call
from pa_copilot.agents.medical_necessity import build_medical_necessity_node
from pa_copilot.schemas import BenefitResult, CriteriaCitation, NecessityAssessment, PARequest


def _state(**kw) -> dict:
    request = PARequest(
        member_id="M100001", service_code="72148", diagnosis_codes=["M54.16"],
        requested_units=1, place_of_service="outpatient", provider_npi="1093817465",
        clinical_summary="MRI lumbar spine", missing_fields=[],
    )
    benefit = BenefitResult(covered=True, plan_id="PPO-100", requires_pa=True, network_status="in_network")
    return {"request": request, "benefit": benefit, "retrieved_criteria": [], **kw}


@tool
def criteria_check(service_code: str, diagnosis_codes: list) -> dict:
    """Check criteria."""
    return {"found": True, "policy_id": "PA-MRI-LUMBAR", "status": "indeterminate",
            "required_conditions": ["8 weeks conservative therapy"], "exclusions": []}


@pytest.mark.asyncio
async def test_medical_necessity_calls_rag_when_indeterminate():
    citation = CriteriaCitation(source="rag_corpus", clause_id="c1", quote="8 weeks PT required", relevance="high")
    expected = NecessityAssessment(criteria_status="met", policy_id="PA-MRI-LUMBAR",
                                    citations=[citation], unmet_requirements=[], confidence=0.8,
                                    rationale="PT documented")
    model = FakeToolCallingModel(
        script=[
            ai_tool_call("criteria_check", {"service_code": "72148", "diagnosis_codes": ["M54.16"]}),
            ai_tool_call("search_clinical_guidance", {"query": "8 weeks PT", "service_code": "72148"}),
            AIMessage(content="assessed"),
        ],
        structured_responses=[expected],
    )
    node = build_medical_necessity_node(mcp_tools=[criteria_check], model=model)
    update = await node(_state())
    assert update["necessity"] == expected
    assert update["retrieved_criteria"] == [citation]


@pytest.mark.asyncio
async def test_medical_necessity_tool_failure_sets_needs_replan():
    @tool
    def broken_criteria_check(service_code: str, diagnosis_codes: list) -> dict:
        """Broken."""
        raise RuntimeError("mcp timeout")

    model = FakeToolCallingModel(
        script=[ai_tool_call("criteria_check", {"service_code": "72148", "diagnosis_codes": ["M54.16"]})],
        structured_responses=[],
    )
    node = build_medical_necessity_node(mcp_tools=[broken_criteria_check], model=model)
    update = await node(_state())
    assert update["needs_replan"] is True
```

> Verify the fake `criteria_check`/`broken_criteria_check` tools are named exactly
> `"criteria_check"` (per PR5a Task 8's naming lesson) — check with `@tool("criteria_check")`
> override syntax if the plain decorator doesn't produce that name from a differently-named
> Python function.

- [ ] **Step 2: Run it red**

Run: `python -m pytest tests/test_agent_medical_necessity.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# src/pa_copilot/agents/medical_necessity.py
"""Medical-necessity worker (design.md §3.3, §4.1, §7): calls MCP criteria_check,
decides — agentically, via the system prompt teaching design.md §4.1's status
mapping, not a hard code gate — whether to call the agentic RAG tool, emits a
validated NecessityAssessment."""

from __future__ import annotations

from typing import Awaitable, Callable

from langchain_core.messages import HumanMessage

from pa_copilot.agents._react import (
    WorkerOutputError,
    WorkerRecursionError,
    WorkerToolError,
    get_agent_model,
    run_worker_react,
    tool_failure_update,
)
from pa_copilot.context.assembly import select_for
from pa_copilot.rag.tool import search_clinical_guidance
from pa_copilot.schemas import NecessityAssessment
from pa_copilot.state import PACaseState

_SYSTEM_PROMPT = (
    "You are the medical-necessity worker for a prior-authorization copilot. Call "
    "criteria_check to get the mechanical policy screen. Its `status` field means: "
    "not_found -> no policy exists, you cannot mechanically assess, treat as "
    "indeterminate with policy_id=None; excluded -> a diagnosis is a documented "
    "exclusion, a strong signal but you should still confirm against the clinical "
    "summary before finalizing not_met; indeterminate -> a policy exists but only "
    "narrative clinical guidance can verify the conditions. Call "
    "search_clinical_guidance whenever criteria_check's status is not_found or "
    "indeterminate, or when there are unmet requirements that need narrative "
    "interpretation, or when you need to confirm an exclusion. Do not call it for a "
    "clearly met case with no unmet requirements. Emit a NecessityAssessment."
)


def build_medical_necessity_node(
    *, mcp_tools: list | None = None, model=None
) -> Callable[[PACaseState], Awaitable[dict]]:
    tools = [*(mcp_tools or []), search_clinical_guidance]

    async def _node(state: PACaseState) -> dict:
        view = select_for("medical_necessity", state)
        request, benefit = view["request"], view["benefit"]
        prompt = HumanMessage(
            content=(
                f"service_code={request.service_code} diagnosis_codes={request.diagnosis_codes} "
                f"clinical_summary={request.clinical_summary!r} covered={benefit.covered} "
                f"requires_pa={benefit.requires_pa}"
            )
        )
        try:
            _messages, necessity = await run_worker_react(
                model or get_agent_model(),
                tools,
                system_prompt=_SYSTEM_PROMPT,
                messages=[prompt],
                response_format=NecessityAssessment,
            )
        except WorkerToolError as exc:
            return tool_failure_update(exc)
        except (WorkerOutputError, WorkerRecursionError):
            return {"needs_replan": True}

        return {"necessity": necessity, "retrieved_criteria": necessity.citations}

    return _node
```

- [ ] **Step 4: Run it green + lint + fast suite + commit**

Run: `ruff check src tests && python -m pytest -q -m "not slow"`

```bash
git add src/pa_copilot/agents/medical_necessity.py tests/test_agent_medical_necessity.py
git commit -m "feat(agents): medical_necessity worker — agentic RAG decision (AC-11 groundwork)"
```

---

### Task 3: `agents/decision_draft.py`

**Files:**
- Create: `src/pa_copilot/agents/decision_draft.py`
- Test: `tests/test_agent_decision_draft.py` (create)

**Interfaces:**
- `build_decision_draft_node(*, store: PolicyStore, model=None) -> node callable`.
  `view = select_for("decision_draft", state)` (`request`, `benefit`, `necessity` —
  never `raw_provider_text`, already enforced by PR5a's selector). Build a `HumanMessage`
  summarizing all three. `run_worker_react(model or get_agent_model(), tools=[],
  system_prompt=..., messages=[prompt], response_format=PADecision)` (no tools — pure
  synthesis). **Self-critique**: for every `citation in decision.cited_criteria`, assert
  `citation.quote` appears in `{c.quote for c in state.get("retrieved_criteria", [])}` —
  if any citation's quote is not found there, return `{"needs_replan": True}` WITHOUT
  setting `state["decision"]` (design.md §3.5: "decision_draft self-critique failure ->
  back to medical_necessity" — the *enforcement* of that loop is PR6's reflection.py; this
  task only detects the mismatch and raises the flag, it does not build the retry). On a
  clean self-critique: if `decision.disposition == "deny"`, write a **critical**-importance
  memory record directly via `store.put(("pa", "member", state["member_id"]), <a fresh
  key, e.g. f"decision:{state['case_id']}">, {"content": <a short summary of the
  determination — disposition, policy_id, reviewer_summary>, "importance": "critical"})`
  — a direct `PolicyStore.put()` call, not the generic `manage_memory` tool (design
  decision from PR5a's `docs/BUILD_LOG.md` handoff: LangMem's tool hardcodes `routine`).
  For any other disposition, skip the memory write for this task (no requirement to write
  routine memories here — design.md §6.1 only specifies "decision_draft writes the
  determination at close" without mandating every disposition writes; keep this task
  scoped to the explicit "denials/appeals = critical" case design.md §6.3 names — if you
  judge a routine write for `approve`/`refer_clinical_review` is clearly intended too,
  note it as a self-review concern rather than silently expanding scope). Return
  `{"decision": decision}` on success (plus nothing else if no memory write happened, or
  nothing extra to report from the memory write either — it's a side effect, not a state
  update).
- `WorkerOutputError`/`WorkerRecursionError` handling: same pattern as Task 2 —
  `{"needs_replan": True}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent_decision_draft.py
import pytest
from langchain_core.messages import AIMessage

from _fakes import FakeToolCallingModel
from pa_copilot.agents.decision_draft import build_decision_draft_node
from pa_copilot.schemas import BenefitResult, CriteriaCitation, NecessityAssessment, PADecision, PARequest


def _state(retrieved_criteria) -> dict:
    request = PARequest(member_id="M100001", service_code="72148", diagnosis_codes=["M54.16"],
                         requested_units=1, place_of_service="outpatient", provider_npi="1093817465",
                         clinical_summary="MRI lumbar spine", missing_fields=[])
    benefit = BenefitResult(covered=True, plan_id="PPO-100", requires_pa=True, network_status="in_network")
    necessity = NecessityAssessment(criteria_status="met", policy_id="PA-MRI-LUMBAR",
                                     citations=retrieved_criteria, unmet_requirements=[],
                                     confidence=0.9, rationale="documented")
    return {"case_id": "case-0001", "member_id": "M100001", "request": request, "benefit": benefit,
            "necessity": necessity, "retrieved_criteria": retrieved_criteria}


@pytest.mark.asyncio
async def test_decision_draft_happy_path_approve(memory_store):
    citation = CriteriaCitation(source="rag_corpus", clause_id="c1", quote="8 weeks PT documented", relevance="high")
    decision = PADecision(disposition="approve", cited_criteria=[citation],
                           reviewer_summary="criteria met", confidence=0.85, human_review_required=True)
    model = FakeToolCallingModel(script=[AIMessage(content="drafted")], structured_responses=[decision])
    node = build_decision_draft_node(store=memory_store, model=model)
    update = await node(_state([citation]))
    assert update["decision"] == decision


@pytest.mark.asyncio
async def test_decision_draft_citation_mismatch_sets_needs_replan(memory_store):
    real_citation = CriteriaCitation(source="rag_corpus", clause_id="c1", quote="8 weeks PT documented", relevance="high")
    hallucinated = CriteriaCitation(source="rag_corpus", clause_id="c2", quote="a quote that was never retrieved", relevance="high")
    decision = PADecision(disposition="approve", cited_criteria=[hallucinated],
                           reviewer_summary="criteria met", confidence=0.85)
    model = FakeToolCallingModel(script=[AIMessage(content="drafted")], structured_responses=[decision])
    node = build_decision_draft_node(store=memory_store, model=model)
    update = await node(_state([real_citation]))
    assert update.get("needs_replan") is True
    assert "decision" not in update


@pytest.mark.asyncio
async def test_decision_draft_deny_writes_critical_memory(memory_store):
    decision = PADecision(disposition="deny", cited_criteria=[], reviewer_summary="not met",
                           confidence=0.7, human_review_required=True)
    model = FakeToolCallingModel(script=[AIMessage(content="drafted")], structured_responses=[decision])
    node = build_decision_draft_node(store=memory_store, model=model)
    await node(_state([]))
    items = memory_store.search(("pa", "member", "M100001"), limit=10)
    assert any(i.value.get("importance") == "critical" for i in items)
```

- [ ] **Step 2: Run it red**

Run: `python -m pytest tests/test_agent_decision_draft.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# src/pa_copilot/agents/decision_draft.py
"""Decision-draft worker (design.md §3.3, §3.5, §6.3): synthesizes benefit +
necessity into a PADecision, self-critiques every cited quote against
retrieved_criteria (a hallucinated citation sets needs_replan rather than being
accepted — PR6's reflection.py owns the actual reroute-and-retry enforcement),
and records a critical-importance memory on denial (design.md §6.3:
"denials/appeals = critical") via a direct PolicyStore.put — LangMem's generic
manage_memory tool hardcodes routine and cannot express this."""

from __future__ import annotations

from typing import Awaitable, Callable

from langchain_core.messages import HumanMessage

from pa_copilot.agents._react import (
    WorkerOutputError,
    WorkerRecursionError,
    get_agent_model,
    run_worker_react,
)
from pa_copilot.context.assembly import select_for
from pa_copilot.memory.store import PolicyStore
from pa_copilot.schemas import PADecision
from pa_copilot.state import PACaseState

_SYSTEM_PROMPT = (
    "You are the decision-draft worker for a prior-authorization copilot. Given the "
    "benefit check and medical-necessity assessment, draft a PADecision. Every quote "
    "in cited_criteria MUST be copied verbatim from the retrieved criteria you were "
    "given — never invent or paraphrase a quote. human_review_required should almost "
    "always be True; this is decision support, not a final determination."
)


def _citations_supported(decision: PADecision, retrieved_criteria: list) -> bool:
    known_quotes = {c.quote for c in retrieved_criteria}
    return all(c.quote in known_quotes for c in decision.cited_criteria)


def build_decision_draft_node(
    *, store: PolicyStore, model=None
) -> Callable[[PACaseState], Awaitable[dict]]:
    async def _node(state: PACaseState) -> dict:
        view = select_for("decision_draft", state)
        request, benefit, necessity = view["request"], view["benefit"], view["necessity"]
        prompt = HumanMessage(
            content=(
                f"service_code={request.service_code} covered={benefit.covered} "
                f"criteria_status={necessity.criteria_status} "
                f"unmet_requirements={necessity.unmet_requirements} "
                f"retrieved_criteria={[c.model_dump() for c in state.get('retrieved_criteria', [])]}"
            )
        )
        try:
            _messages, decision = await run_worker_react(
                model or get_agent_model(),
                tools=[],
                system_prompt=_SYSTEM_PROMPT,
                messages=[prompt],
                response_format=PADecision,
            )
        except (WorkerOutputError, WorkerRecursionError):
            return {"needs_replan": True}

        if not _citations_supported(decision, state.get("retrieved_criteria", [])):
            return {"needs_replan": True}

        if decision.disposition == "deny":
            store.put(
                ("pa", "member", state["member_id"]),
                f"decision:{state['case_id']}",
                {
                    "content": (
                        f"Prior-auth denied for service {request.service_code}: "
                        f"{decision.reviewer_summary}"
                    ),
                    "importance": "critical",
                },
            )

        return {"decision": decision}

    return _node
```

- [ ] **Step 4: Run it green + lint + fast suite + commit**

Run: `ruff check src tests && python -m pytest -q -m "not slow"`

```bash
git add src/pa_copilot/agents/decision_draft.py tests/test_agent_decision_draft.py
git commit -m "feat(agents): decision_draft worker — self-critique + critical memory on deny"
```

---

### Task 4: `agents/human_review.py`

**Files:**
- Create: `src/pa_copilot/agents/human_review.py`
- Test: `tests/test_agent_human_review.py` (create)

**Interfaces:**
- `build_human_review_node() -> Callable[[PACaseState], Awaitable[dict]]` — the returned
  node calls `langgraph.types.interrupt({"case_id": ..., "request": ...,
  "necessity": ..., "decision": ..., "reason": "human review required"})` (build the
  payload from whatever's already in state — a small dict, not the full state; use
  `.model_dump()` on any Pydantic fields present so the payload is JSON-shaped) and
  returns `{"working_memory": write_working_memory(state, "human_review_resume_value",
  value)["working_memory"]}` where `value` is `interrupt(...)`'s return (the resume
  value the caller supplies later) — recording it is optional polish, not a hard
  requirement; the main point of this task is that the node genuinely pauses the graph.
  A **top-level** test of `interrupt()` pausing/resuming needs a compiled graph with a
  real checkpointer (Task 6), so this task's own test can only verify the node function's
  shape in isolation — see Step 1's note.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_human_review.py
"""human_review is a thin interrupt() stub — genuinely testing that it PAUSES a
graph run needs a compiled graph + real checkpointer (see test_ac05_checkpointer.py,
Task 6). This test only proves the node calls interrupt() with a sane payload; it
must be exercised from inside a langgraph node execution context (interrupt() reads
an ambient contextvar), so drive it through a minimal throwaway StateGraph rather
than calling the node function bare."""

import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from pa_copilot.agents.human_review import build_human_review_node


@pytest.mark.asyncio
async def test_human_review_interrupts_then_resumes():
    node = build_human_review_node()
    g = StateGraph(dict)
    g.add_node("human_review", node)
    g.add_edge(START, "human_review")
    g.add_edge("human_review", END)
    compiled = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-test"}}

    first = await compiled.ainvoke({"case_id": "case-0001"}, config=cfg)
    assert "__interrupt__" in first

    second = await compiled.ainvoke(Command(resume="approved-by-reviewer"), config=cfg)
    assert "__interrupt__" not in second
```

> `langgraph.checkpoint.memory.InMemorySaver` is a real, synchronous-and-async-capable
> in-process checkpointer meant exactly for tests like this — verify its import path
> against the installed `langgraph` version if this doesn't resolve (it may live at
> `langgraph.checkpoint.memory.InMemorySaver` or a similarly-named class in that module;
> check `python -c "import langgraph.checkpoint.memory as m; print(dir(m))"` if unsure).

- [ ] **Step 2: Run it red**

Run: `python -m pytest tests/test_agent_human_review.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# src/pa_copilot/agents/human_review.py
"""human_review (design.md §3.3): terminal stub representing handoff to a human
reviewer. interrupt() pauses graph execution; the checkpointer persists the state
so the case can be resumed later, in another process (AC-05)."""

from __future__ import annotations

from typing import Awaitable, Callable

from langgraph.types import interrupt

from pa_copilot.context.assembly import write_working_memory
from pa_copilot.state import PACaseState


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
        return write_working_memory(state, "human_review_resume_value", resume_value)

    return _node
```

- [ ] **Step 4: Run it green + lint + fast suite + commit**

Run: `ruff check src tests && python -m pytest -q -m "not slow"`

```bash
git add src/pa_copilot/agents/human_review.py tests/test_agent_human_review.py
git commit -m "feat(agents): human_review — interrupt() stub for the pause/resume handoff"
```

---

### Task 5: `graph.py` — the hand-rolled `StateGraph`

**Files:**
- Create: `src/pa_copilot/graph.py`
- Test: `tests/test_graph_topology.py`, `tests/test_ac01_typed_state.py` (create)

**Interfaces:**
- `async def make_graph(*, store: PolicyStore, checkpointer, mcp_tools: list, model=None)
  -> CompiledStateGraph` — assembles the topology exactly per design.md §3.1:
  ```
  START -> summarize -> supervisor
  supervisor --conditional on state["next"]--> {intake|benefit_check|medical_necessity
                                                 |decision_draft|human_review|END}
  intake, benefit_check, medical_necessity, decision_draft, human_review -> summarize -> supervisor
  ```
  Node names: `"summarize"`, `"supervisor"`, `"intake"`, `"benefit_check"`,
  `"medical_necessity"`, `"decision_draft"`, `"human_review"`. Build each worker node via
  its `build_X_node(...)` factory, passing `store=` (intake, decision_draft) and
  `mcp_tools=` (filtered to the specific tools each worker needs, or pass the full list —
  each worker only calls the tools its own system prompt asks the model to call; passing
  extra unused tools is harmless but check whether the plan's earlier tasks assumed a
  filtered list before deciding). `add_conditional_edges("supervisor", lambda s:
  s["next"], {"intake": "intake", "benefit_check": "benefit_check", "medical_necessity":
  "medical_necessity", "decision_draft": "decision_draft", "human_review": "human_review",
  "FINISH": END})`. Every worker node has a plain `add_edge(worker, "summarize")`.
  `add_edge("summarize", "supervisor")`, `add_edge(START, "summarize")`. Compile with
  `checkpointer=checkpointer, store=store`.
- `graph.get_graph().draw_ascii()` — used by `scripts/graph_topology_demo.py` (Task 6),
  not inline here.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph_topology.py
"""Structural assertions on the compiled graph — every node design.md §3.1 names
is present, the conditional edge's path_map covers all 6 targets."""

import pytest

from pa_copilot.graph import make_graph
from _fakes import FakeToolCallingModel


@pytest.mark.asyncio
async def test_graph_has_all_seven_nodes(memory_store):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = await make_graph(
        store=memory_store, checkpointer=InMemorySaver(), mcp_tools=[],
        model=FakeToolCallingModel(),
    )
    node_names = set(graph.get_graph().nodes.keys())
    expected = {"__start__", "summarize", "supervisor", "intake", "benefit_check",
                "medical_necessity", "decision_draft", "human_review", "__end__"}
    assert expected.issubset(node_names)
```

```python
# tests/test_ac01_typed_state.py
"""AC-01: PACaseState is an explicit typed state object shared across nodes."""

import typing

from pa_copilot.state import PACaseState


def test_pa_case_state_is_a_typed_dict():
    assert typing.is_typeddict(PACaseState)


def test_every_node_module_reads_and_writes_pa_case_state():
    # Structural check: each worker/supervisor/summarize module's node callable
    # is a plain async function accepting one PACaseState-shaped dict and
    # returning a dict — verified by this PR's own worker tests already
    # exercising that contract; this test asserts the modules import cleanly
    # and expose the expected factory/function names.
    from pa_copilot import supervisor
    from pa_copilot.agents import benefit_check, decision_draft, human_review, intake, medical_necessity
    from pa_copilot.context import summarization

    assert hasattr(supervisor, "build_supervisor_node")
    assert hasattr(intake, "build_intake_node")
    assert hasattr(benefit_check, "build_benefit_check_node")
    assert hasattr(medical_necessity, "build_medical_necessity_node")
    assert hasattr(decision_draft, "build_decision_draft_node")
    assert hasattr(human_review, "build_human_review_node")
    assert hasattr(summarization, "summarize")
```

- [ ] **Step 2: Run it red**

Run: `python -m pytest tests/test_graph_topology.py tests/test_ac01_typed_state.py -v`
Expected: FAIL (`ModuleNotFoundError: pa_copilot.graph`).

- [ ] **Step 3: Implement `graph.py`**

Build it from the Interfaces block above — this is assembly, not new logic; every
component it wires already exists and is tested. Decide the `mcp_tools` filtering
question (pass the full list vs. a per-worker subset) empirically: try passing the full
list to every worker first (simplest), and only split it per-worker if a test reveals a
problem (e.g. a worker's system prompt not mentioning an irrelevant tool could still let
the model call it accidentally — if you see that in a test, split the list: `intake`
gets `provider_lookup`-shaped tools, `benefit_check` gets `benefit_lookup`-shaped,
`medical_necessity` gets `criteria_check`-shaped, filtered by tool `.name`).

- [ ] **Step 4: Run it green + lint + fast suite + commit**

Run: `ruff check src tests && python -m pytest -q -m "not slow"`

```bash
git add src/pa_copilot/graph.py tests/test_graph_topology.py tests/test_ac01_typed_state.py
git commit -m "feat(graph): hand-rolled StateGraph topology (AC-01 groundwork)"
```

---

### Task 6: Checkpointer wiring + AC-05 pause/resume evidence

**Files:**
- Create: `scripts/graph_topology_demo.py`, `scripts/run_pause_resume_test.py`
- Test: `tests/test_ac05_checkpointer.py` (create)
- Modify: `Makefile` (add a `pause-resume-test` target, mirroring PR4's
  `persistence-test` target)

**Interfaces:**
- `scripts/graph_topology_demo.py` — builds a `make_graph(...)` with
  `FakeToolCallingModel` + an in-memory store/checkpointer (no real files needed — this
  is a structural dump, not a run), writes `graph.get_graph().draw_ascii()` to
  `traces/graph_topology.txt` (byte-stable: no run-specific IDs in `draw_ascii()`'s
  output — verify this empirically; if `draw_ascii()` embeds anything non-deterministic,
  note it and use `draw_mermaid()` instead, updating design.md's note accordingly).
- `scripts/run_pause_resume_test.py` — mirrors `scripts/run_persistence_test.py`'s shape
  (PR4): spawns two separate `python -c "..."` (or a small helper script invoked twice)
  subprocess invocations sharing one `.pa_state.db` (and a throwaway `.pa_memory.db`, or
  reuse the same file — your call). First invocation: build the real graph via
  `make_graph(...)` (fakes for the model, since no API key), run a sample case to
  `human_review`'s interrupt; assert `"__interrupt__"` present; print progress to
  stdout. Second invocation: build a **fresh** graph object (new process, new
  connections) against the same files, resume with `Command(resume=<value>)`, assert
  completion. Combined stdout → `traces/pause_resume_transcript.md` (committed).
- `tests/test_ac05_checkpointer.py` — the deterministic, in-process version of the same
  proof (teardown + rebuild in-process, mirroring PR4's
  `tests/test_memory_persistence.py` two-forms pattern): build a graph, run to
  interrupt, tear down the graph/connection object (but NOT the underlying `tmp_path`
  db file), build a **new** graph object from the same file, resume, assert completion.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ac05_checkpointer.py
"""AC-05: a checkpointer persists graph state so a case can be paused and
resumed. In-process teardown+rebuild form (genuine 2-process form is
scripts/run_pause_resume_test.py -> traces/pause_resume_transcript.md)."""

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
import aiosqlite

from pa_copilot.graph import make_graph
from pa_copilot.memory.store import memory_store
from _fakes import FakeToolCallingModel


@pytest.mark.asyncio
async def test_interrupt_then_resume_from_a_fresh_graph_object(tmp_path, fake_embedder):
    db_path = str(tmp_path / "state.db")
    thread = {"configurable": {"thread_id": "case-ac05"}}

    with memory_store(tmp_path / "mem.db", embedder=fake_embedder) as store:
        conn1 = await aiosqlite.connect(db_path)
        saver1 = AsyncSqliteSaver(conn1)
        await saver1.setup()
        graph1 = await make_graph(store=store, checkpointer=saver1, mcp_tools=[],
                                   model=FakeToolCallingModel())
        # ... drive graph1 to human_review's interrupt with a scripted case ...
        # (fill in the exact initial state / fake script needed to route straight
        # to human_review deterministically via hard_route, e.g. supervisor_hops
        # forced past max_hops, or a decision already set with needs_replan=True
        # cleared, or the simplest deterministic path you can construct)
        await conn1.close()

        conn2 = await aiosqlite.connect(db_path)
        saver2 = AsyncSqliteSaver(conn2)
        graph2 = await make_graph(store=store, checkpointer=saver2, mcp_tools=[],
                                   model=FakeToolCallingModel())
        result = await graph2.ainvoke(Command(resume="approved"), config=thread)
        assert "__interrupt__" not in result
        await conn2.close()
```

> This test's exact "drive graph1 to human_review deterministically" step needs real
> design work once you're implementing it — the simplest reliable path is probably to
> seed `state["supervisor_hops"]` at/above `max_hops` so `hard_route`'s hop-cap guardrail
> forces `human_review` on the very first supervisor turn, skipping the need to script
> every worker's LLM call. Verify this actually reaches `human_review` before relying on
> it; if the hop-cap path doesn't cleanly reach human_review (e.g. `summarize` or some
> other node runs first and needs its own script entry), adjust.

- [ ] **Step 2: Run it red**

Run: `python -m pytest tests/test_ac05_checkpointer.py -v`
Expected: FAIL (incomplete — you're filling in the driving logic as you implement).

- [ ] **Step 3: Implement the two scripts + finish the test**

Follow PR4's `scripts/run_persistence_test.py` as the structural template for
`run_pause_resume_test.py` (two subprocess invocations, combined stdout, byte-stable
committed log). `graph_topology_demo.py` is simpler — no subprocess needed.

- [ ] **Step 4: Run everything green, generate the real committed traces**

Run: `python scripts/graph_topology_demo.py` → `traces/graph_topology.txt`
Run: `python scripts/run_pause_resume_test.py` → `traces/pause_resume_transcript.md`
Run: `python -m pytest tests/test_ac05_checkpointer.py -v`

- [ ] **Step 5: Lint + full fast suite + commit**

Run: `ruff check src tests scripts && python -m pytest -q -m "not slow"`

```bash
git add scripts/graph_topology_demo.py scripts/run_pause_resume_test.py Makefile \
        tests/test_ac05_checkpointer.py traces/graph_topology.txt \
        traces/pause_resume_transcript.md
git commit -m "feat(graph): AsyncSqliteSaver wiring + AC-05 pause/resume evidence"
```

---

### Task 7: AC-02/03/10/11 full in-graph evidence

**Files:**
- Create: `scripts/full_case_demo.py`
- Test: `tests/test_ac02_supervisor_routing.py`, `tests/test_ac03_conditional_routing.py`,
  `tests/test_ac10_mcp_integration.py`, extend `tests/test_ac11_agentic_rag.py` (create/modify)

**Interfaces:**
- `scripts/full_case_demo.py` — runs two full cases through the real `make_graph(...)`
  (fakes throughout — deterministic, no API key) to completion (`FINISH`, not
  `human_review`): (a) a clear-cut case that reaches `decision_draft` via ≥2 workers,
  producing `traces/run_full_case.json` (via `RunTracer`, wired as a thin observer —
  either pass a tracer callback into each node or, simpler, just serialize `route_history`
  + the final `decision`/`necessity`/`benefit`/`request` from the graph's final state
  dict directly, whichever is more consistent with `tracing.py`'s existing shape — your
  call, but keep it consistent with `RunTracer`'s schema if you use it, since
  `test_nfr04_trace_schema.py` (PR7) will eventually validate every `traces/*.json`); (b)
  an ambiguous/low-confidence case that routes to `human_review` instead of an auto-draft,
  producing `traces/route_ambiguous.json`; the clear-cut case's routing trail also becomes
  `traces/route_clearcut.json` (same run, different extraction, or a third dedicated
  script pass — your call on whether one run produces both files or two separate runs
  do).
- `tests/test_ac02_supervisor_routing.py` — asserts a full run's `route_history` visits
  ≥2 distinct workers.
- `tests/test_ac03_conditional_routing.py` — same compiled graph, two different states:
  clear-cut → reaches `decision_draft`; ambiguous → reaches `human_review`.
- `tests/test_ac10_mcp_integration.py` — a run where `medical_necessity` actually invokes
  `criteria_check` (through `mcp_client.load_pa_tools`, not a fake MCP tool) — this one
  legitimately needs the real MCP server subprocess (already exercised in PR2/PR3's own
  tests) even though the LLM is faked; use `pa_session()`/`load_pa_tools(session=...)`
  from `mcp_client.py`.
- `tests/test_ac11_agentic_rag.py` — extend PR3's file (or add a new one alongside it) —
  a full in-graph run where `medical_necessity` calls `search_clinical_guidance` for an
  indeterminate case, and a second run where a clear `met`/`excluded` case does not call
  it — proving the *agent* decides, inside the compiled graph, not just at the tool level
  (PR3 only proved the tool-level predicate).

Add a `@pytest.mark.slow + skipif(not GEMINI_API_KEY)` counterpart test for at least one
full case, per this project's established pattern for evidence that ideally gets
regenerated with a real key later.

- [ ] **Step 1: Write the failing tests** (design the exact deterministic state/script
  needed for a clear-cut vs. ambiguous case — reuse the `sample_request` fixture's shape,
  member `M100001`/service `72148` for clear-cut; construct a second case with a service
  code that maps to `criteria_check` returning `indeterminate` with no clean RAG match, or
  force ambiguity via a low `confidence` in a scripted `NecessityAssessment`, whichever is
  simpler to script reliably)

- [ ] **Step 2: Run red**

- [ ] **Step 3: Implement `scripts/full_case_demo.py`, run it, generate the traces**

- [ ] **Step 4: Run green + lint + full fast suite + commit**

```bash
git add scripts/full_case_demo.py tests/test_ac02_supervisor_routing.py \
        tests/test_ac03_conditional_routing.py tests/test_ac10_mcp_integration.py \
        tests/test_ac11_agentic_rag.py traces/run_full_case.json \
        traces/route_clearcut.json traces/route_ambiguous.json
git commit -m "feat(graph): full in-graph run evidence — AC-02/03/10/11"
```

---

### Task 8: AC-04 + NFR-03 full closure

**Files:**
- Create: `tests/test_ac04_structured_output.py`, extend `tests/test_nfr03_quarantine.py`
  (or create the full-closure version alongside PR5a's structural one — check whether
  PR5a's `tests/test_agent_intake.py` canary test already satisfies this file name;
  design.md/specs name `tests/test_nfr03_quarantine.py` specifically, which does not yet
  exist as its own file — PR5a's canary test lives in `test_agent_intake.py` instead;
  decide whether to rename/move or add a new file satisfying the exact name the ledger
  promises, and update `specs/nfr.md`'s test-file reference if you keep it where it is)

**Interfaces:**
- `tests/test_ac04_structured_output.py` — for at least 2 workers, feed a
  `FakeToolCallingModel` a `structured_responses` entry that is NOT a valid instance of
  the expected schema (e.g. missing a required field) and assert
  `pydantic.ValidationError` surfaces as `WorkerOutputError` (PR5a's fix wave), not a
  silent pass-through.
- Full NFR-03 closure: drive the **real compiled graph** with the injection-canary raw
  text (design.md §5.1's exact canary language) through to a final `decision`, and assert
  the disposition is not coerced to `"approve"` by the injected instruction — this is the
  test PR5a's plan deferred here because it needs `decision_draft` to exist.

- [ ] **Step 1: Write the failing tests**
- [ ] **Step 2: Run red**
- [ ] **Step 3: Implement/wire whatever's needed to make them pass** (likely nothing new
  in `src/` — this task is about proving existing behavior, not adding new mechanism,
  unless a real gap turns up)
- [ ] **Step 4: Run green + lint + full fast suite + commit**

```bash
git add tests/test_ac04_structured_output.py tests/test_nfr03_quarantine.py \
        traces/quarantine_canary.json
git commit -m "test: AC-04 + NFR-03 full closure evidence"
```

---

### Task 9: Ledger updates + `design.md` correction + whole-branch pass

**Files:**
- Modify: `specs/acceptance-criteria.md`, `specs/nfr.md`, `docs/design.md`

**Steps:**

- [ ] **Step 1: Update `specs/acceptance-criteria.md`** — AC-01, AC-02, AC-03, AC-04,
  AC-05 rows → `done`, each with a short PR5b evidence note (test file + trace). AC-10
  row → `done` (full in-graph proof; the representative-transcript note from PR2 stays
  for the historical record). AC-11 row → `done` (full in-graph "agent decides" proof;
  PR3's tool-level note stays for the historical record).

- [ ] **Step 2: Update `specs/nfr.md`** — NFR-03 row → `done`, full "disposition not
  forced" proof + `traces/quarantine_canary.json`. NFR-04 row note: traces/ now has
  schema-shaped evidence from a real graph run (still `partial` overall — full trace
  schema validation is PR7's `test_nfr04_trace_schema.py`).

- [ ] **Step 3: Correct `docs/design.md` §1's stack table** — `SqliteSaver` →
  `AsyncSqliteSaver`, one-line note on why (the sync one cannot run any async
  checkpoint operation at all, verified empirically this PR — see this plan's Library
  facts section for the full writeup, worth summarizing briefly in design.md itself so a
  future reader doesn't reach for the wrong class).

- [ ] **Step 4: Full whole-branch pass**

Run: `ruff check src tests scripts && python -m pytest -q -m "not slow"`
Run: `python -m pytest -q` (include slow, confirm nothing unexpectedly marked slow)

- [ ] **Step 5: Commit**

```bash
git add specs/acceptance-criteria.md specs/nfr.md docs/design.md
git commit -m "docs: PR5b ledger updates — AC-01/02/03/04/05/10/11 + NFR-03 done, SqliteSaver correction"
```

---

## After this branch: handoff to PR6

Update `docs/BUILD_LOG.md` (mirror PR1-5a's entries) once merged. PR6 (`feat/reflection`)
scope per design.md §10: `reflection.py`, the dual-trigger reflection/self-healing loop
(tool failure + low confidence — design.md §3.5), `tenacity` retries, `asyncio.wait_for`
timeouts, `MAX_REPLANS`/`MAX_HOPS` enforcement (currently unenforced — `replan_count` is
never incremented anywhere in PR5a/PR5b), and `agents/_react.py::_best_effort_tool_name`'s
real tool-attribution gap (currently `"unknown_tool"` for any worker bound to 2+ tools —
flagged as PR6's first item in PR5a's handoff, still true after PR5b since
`medical_necessity` also binds 2 tools).
