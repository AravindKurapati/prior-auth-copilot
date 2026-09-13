# Build Log — Prior-Authorization Copilot

Per-PR record of what shipped and the decisions behind it. The authoritative spec is
`docs/design.md`; per-PR task plans are `docs/implementation-plan-pr{N}.md`; the AC/NFR
ledgers are `specs/`.

## Method

- **PR-driven, `git merge --no-ff` per PR**, no direct commits to `main` after each PR's
  plan commit. First-parent history = scaffold → (plan → merge) × N.
- Each PR: write `docs/implementation-plan-pr{N}.md` → execute task-by-task under the
  `superpowers:subagent-driven-development` skill (fresh implementer per task/batch, a task
  review after each, a whole-branch review on the strongest model before merge, one fix
  wave + one scoped re-review, then merge via `superpowers:finishing-a-development-branch`).
- Local `--no-ff` merges to `main` are pre-authorized for this workflow. **As of PR6's
  merge, pushed to a public GitHub remote**: https://github.com/AravindKurapati/prior-auth-copilot
  (`origin/main`, full existing history pushed as-is, no rewrite). PR1-6 were still
  built entirely via local `--no-ff` merges with no remote involved — the remote only
  now exists as a mirror/backup and an option for future PRs. Whether PR7+ open real
  GitHub Pull Requests or keep merging locally-then-pushing is a per-PR call the user
  makes, not a default — ask before assuming either way.
- SDD ledgers with every ruling live in `.superpowers/sdd/implementation-plan-pr{N}/progress.md`
  (gitignored). Retained across the whole build for recovery + the rulings summary.

## Standing constraints (carry forward to every PR)

- **No `GEMINI_API_KEY` in the build environment.** Live-Gemini tests are
  `@pytest.mark.slow` + `@pytest.mark.skipif(not GEMINI_API_KEY)`. Committed traces that
  need an LLM are **representative** (real tool/retrieval outputs, reconstructed agent
  turns, clearly labelled with a regenerate-with-a-key note). The user regenerates the
  genuine ones on a machine with a key. `config.load_settings` mirrors
  `GEMINI_API_KEY` → `GOOGLE_API_KEY` (langchain-google-genai reads the latter).
- **The real bge-small model DOES download** in this environment — RAG evidence is genuine.
- **Stack is langchain 0.3.x / langchain-core 0.3.x** (do NOT upgrade to 1.x mid-build).
  Pins: `langgraph>=1.0,<2`, `langgraph-checkpoint-sqlite>=3,<4`, `langchain-mcp-adapters>=0.1.10,<0.2`,
  `mcp>=1.9,<2`, `chromadb>=0.6,<0.7`, `pytest-asyncio>=0.24,<2`. `constraints.txt` pins the
  tested resolution; README quick-start uses `pip install -e ".[dev]" -c constraints.txt`.
- Test output must be **pristine** under `pyproject` `filterwarnings = ["error"]` (+ two
  narrow `chromadb` deprecation ignores). New warnings fail the suite.
- Committed evidence under `traces/` and `data/synthetic/*.jsonl` must be **byte-stable**
  across regeneration (no timestamps in committed artifacts; explicit `encoding="utf-8",
  newline="\n"`).
- Every AC/NFR test + artifact carries its `AC-NN`/`NFR-NN` id. `tests/test_ac_traceability.py`
  fails any `done`/`partial` ledger row that names a test file which doesn't exist.
- Windows: never `shutil.rmtree` a live chromadb client's dir; use `client.delete_collection`.

## PR1 — Foundations & Contracts  (merged `35ee81c`)

Typed `PACaseState` (AC-01), all Pydantic node-boundary schemas + `SampleSubmission`
(AC-04), `config.Settings` with filesystem anchors + frozen `MemoryConfig`, `RunTracer`
with automatic name+member-id redaction + crash-safe degraded traces (NFR-05 done, NFR-04
partial), synthetic corpora + 7 sample requests, warning-trap regression test,
`test_ac_traceability.py` meta-test.

Key: `RunTracer` normalises to JSON primitives *before* redaction (non-primitive payloads
were a leak). `Settings.memory` is a frozen `MemoryConfig` (not a raw dict) for PR4's
per-test isolation. NFR-01 secret-scan regex is narrow (won't flag `<placeholder>` values).

## PR2 — Custom MCP Server + Adapter Integration  (merged `e53b472`)

`mcp_server/` — FastMCP over stdio: 3 tools (`benefit_lookup`, `provider_lookup`,
`criteria_check`) + 2 resources (`pa://criteria/index`, `pa://criteria/{policy_id}`),
backed by pure `data_access` lookups over `data/synthetic/` (AC-09 done, evidence
`traces/mcp_capabilities.json`). `mcp_client.py` — `MultiServerMCPClient` + a
**`pa_session()` persistent-session seam** + `pa_server_spec(env=, cwd=)` so PR5's graph
holds ONE connection and tests can redirect corpora. `criteria_check` status ∈
`{not_found, excluded, indeterminate}` — the MCP→worker status mapping table is in
`docs/design.md` §4.1 (PR5 consumes it). `docs/integration-decision.md` (NFR-06 partial).
AC-10 partial — `traces/mcp_toolcall_transcript.md` is representative (needs a key for the
genuine one).

Key: `data_access` never raises on a miss (returns `{found: False, ...}`) except
`load_corpora` → `CorporaUnavailable`. `_get_tool_embedder`-style module caches are cleared
by the autouse `_clear_settings_cache` fixture.

## PR3 — Agentic RAG Tool  (merged `2d8762a`)

`rag/` — `search_clinical_guidance` LangChain `@tool` (returns `list[dict]` of
`CriteriaCitation.model_dump()`) + `should_search_guidance(criteria_status, unmet_requirements=)`
predicate over an in-process Chroma index (local bge-small) of the 6 committed
`data/synthetic/clinical_guidance/*.md`. Retrieval is **predicate-gated inside the agent
loop**, not a fixed stage (AC-11 partial; full in-graph run PR5). Degrades to `[]` on:
weak retrieval (best hit `< rag_min_score` after one corrective rewrite), stale /
wrong-model / wrong-dimension `.pa_chroma/` (metadata-stamped + checked), or
`RagIndexUnavailable` — so PR5 routes those to `human_review`. `traces/agentic_rag_decision.md`
is **genuine** bge-small retrieval. Build the index with `make ingest-rag` /
`scripts/ingest_rag.py` (`pac ingest` wiring is PR7).

Key: `rag/tool.py` must NOT import `data.synthetic.generators` (repo-root package,
unimportable outside repo root) — it routes `_service_name` through
`data_access.list_policies()`. Guard: `tests/test_rag_tool_importable.py` subprocess-imports
from an outside cwd. `_default_embedder` is a cached module global (built once).

**Parked for PR5:** `tool._service_name` → `list_policies()` → `load_corpora()` can raise
`CorporaUnavailable` (criteria.json missing), NOT caught by the tool's
`except RagIndexUnavailable`. PR5's `medical_necessity` worker should widen its tool-call
try/except to `(RagIndexUnavailable, CorporaUnavailable)` → `[]`.

---

## PR4 — Memory Subsystem  (merged `050b488`)

Tiered memory (design.md §6; AC-06, AC-07, AC-08): Tier-1 `memory/working.py`
(`remember`/`recall`/`search_working` — pure dict transforms over
`state["working_memory"]["facts"]`, checkpointer-persisted starting PR5) + Tier-2
`PolicyStore` (`memory/store.py`, a `SqliteStore` subclass) over `.pa_memory.db`, with an
optional `sqlite-vec` semantic index (local bge-small via `memory/embeddings.py`'s
`LocalEmbeddings`/`as_embeddings` wrapping `rag.embedder.BgeEmbedder`). Namespaces
`("pa","member",<id>)`, `("pa","provider",<npi>)`, `("pa","policy_notes")`,
`("pa","episodic")`. `memory/tools.py::build_memory_tools` wraps langmem's
`create_manage_memory_tool`/`create_search_memory_tool` with an explicit `store=`.
`memory/policy.py` (pure functions plus store-coupled `search_ranked`/`enforce_cap`/
`sweep_expired`) implements per-namespace TTL, `{routine,notable,critical}` importance
weighting, recency half-life decay, and a soft-against-`critical` LRU cap. AC-07 has both an
in-process teardown+rebuild test (`tests/test_memory_persistence.py`) and a genuine
2-process test (`scripts/run_persistence_test.py` → `traces/memory_persistence.log`).
Full policy writeup: `docs/memory-policy.md`.

Key: `TTLConfig` only supports one global `default_ttl`, so `PolicyStore.put`/`aput`
compute and inject a **per-item** TTL (`policy.ttl_minutes_for`) and a default importance on
every write before enforcing the namespace's cap — this is how a single native store
achieves the per-namespace policy design.md §6.3 calls for. `open_memory_store()` degrades
to `index=None` (no sqlite-vec) on extension-load failure, recording `store.semantic_error`
rather than raising, so AC-06/07/08 hold on a machine without `sqlite-vec` available.
Three real bugs were caught and fixed in review, not part of the original design: (1)
`enforce_cap`'s internal namespace-listing scan must pass `refresh_ttl=False` — otherwise
every write's cap-check silently refreshed the TTL of every *other* item in the namespace,
defeating per-item expiry; (2) `open_memory_store` must close the sqlite3 connection on
*both* failure edges before propagating/retrying, or it leaks the handle (observed as a
locked-file failure on Windows); (3) — caught only at the **final whole-branch review**,
invisible to 9 scoped task reviews — `enforce_cap`/`search_ranked` listed a namespace via
`store.search(namespace, ...)`, which does a SQL *prefix* match with no trailing separator,
so `("pa","member","M1")` also matched `("pa","member","M10")`: the cap silently failed to
enforce (deleting a key that belonged to the other namespace no-ops) and `search_ranked`
could return another member's records. Fixed by filtering both functions' results to the
exact namespace tuple before any counting/ranking; regression test covers both directions
and both functions (`tests/test_memory_store.py`). AC-06/07/08 evidence:
`tests/test_ac06_tiered_memory.py` + `traces/tiered_memory_recall.json`;
`tests/test_memory_persistence.py` + `scripts/run_persistence_test.py` +
`traces/memory_persistence.log`; `tests/test_ac08_eviction.py` (its recency-eviction case
backdates one item's `created_at`/`updated_at` via raw SQL, because SQLite's
`CURRENT_TIMESTAMP` only has 1-second granularity and back-to-back test writes otherwise tie
on recency — see `docs/memory-policy.md` §8). 129 tests pass (125 fast + 4 slow, 1 live-Gemini
skip without a key), pristine; `ruff check src tests` clean.

**Known limitation, not fixed here (PR5 watch-out):** LangMem's memory tools only work
through synchronous `.invoke()` against `PolicyStore` — `.ainvoke()` raises
`NotImplementedError`. Verified against the installed `langmem`/`langgraph` source: LangMem's
async tool path does call `await store.aput(...)` directly and reaches `PolicyStore.aput`
fine; the failure is one level deeper — `BaseStore.aput` unconditionally calls
`await self.abatch(...)`, and `SqliteStore.abatch` (the langgraph library's own override)
unconditionally raises for **any** async batch operation. So `PolicyStore.aput` cannot
succeed from any async caller at all — not just LangMem's tool, but a direct first-party
`await store.aput(...)` too. This is a firm, verified fact (it will raise), not an unproven
gap. See §10 of `docs/memory-policy.md`.

## PR5a — Graph Core, Part A  (merged `3f85be8`)

`docs/design.md` §10 allows splitting PR5 when it grows large; approved 2026-09-11.
`context/` — `quarantine.py` (NFR-03 mechanism: `raw_provider_text` isolated to a
user-role `HumanMessage`, HTML-escaped so it can never forge the delimiter tags, a
preamble-checked audit helper), `assembly.py` (`select_for(node, state)` per-node minimal
field table; `write_working_memory`), `summarization.py` (wraps the real
`langmem.short_term.SummarizationNode`, closes **NFR-08** —
`traces/context_before_after.md`, byte-stable). `agents/_react.py` — shared
`get_agent_model()` + `run_worker_react()` built on
`langgraph.prebuilt.create_react_agent(model, tools, response_format=Schema)`
(a real, already-working mechanism: verified against the installed langgraph 1.0.1
source that it makes one separate `model.with_structured_output(schema)` call after the
tool loop, rather than needing a hand-rolled ReAct loop); `ToolNode(handle_tool_errors=
False)` so a real tool exception reaches this module and becomes one uniform
`WorkerToolError` (a `ValidationError` from the structured-output call becomes a distinct
`WorkerOutputError`, and `GraphRecursionError` a `WorkerRecursionError` — not blamed on a
tool). `supervisor.py` — deterministic guardrails (`hard_route`, hop-cap → tool-failure-cap
→ decision-finished → request-None → `None`) + one `.with_structured_output(RouterDecision)`
LLM call, no tools. `agents/intake.py` (quarantined extraction + MCP `provider_lookup` +
member-namespace memory search, emits `PARequest`) and `agents/benefit_check.py` (MCP
`benefit_lookup`, emits `BenefitResult`) — the first two of five workers.
`tests/_fakes.py::FakeToolCallingModel` is the new shared LLM test double: a scripted
`BaseChatModel` driven through the real `create_react_agent` graph (tool call →
`ToolMessage` → final `AIMessage` → separate structured call), so every worker test
exercises real machinery, not a hand-rolled mock. Advances AC-01/02/04 and NFR-03/04
toward PR5b's close (none of these are `done` yet — see `specs/`). Does **not** build
`graph.py`, the checkpointer wiring, or the remaining three workers
(`medical_necessity`, `decision_draft`, `human_review`) — that is PR5b.

Key: `agents/_react.py`'s docstring records what the plan's own research found
empirically (library facts about `create_react_agent`, `ToolNode`'s exception
propagation, `SummarizationNode`'s `state["context"]["running_summary"]` contract,
LangMem's generic `{key}` namespace templating) so PR5b doesn't have to re-derive any of
it. **One real Critical bug, caught only at the final whole-branch review** (invisible to
every task-scoped review, including the batched Task 8+9 review that built the worker
that hit it): `PolicyStore` had no `abatch` override, so `intake`'s bound
member-namespace `search_memory` tool raised `NotImplementedError` on *every* invocation
— `ToolNode`'s async dispatch (required since `run_worker_react` calls `.ainvoke()`)
routes through LangMem's `asearch_memory` → `store.asearch` → `BaseStore.abatch` →
`SqliteStore.abatch`, which unconditionally raises for any async batch op. This aborted
the whole `intake` node and mislabeled the failure as `tool="unknown_tool"`. Root cause:
the PR5a plan's own "Spec" section asserted workers only call memory tools
*synchronously* — false; a tool bound into `create_react_agent`'s loop is dispatched by
`ToolNode` via whatever path the graph's own `.ainvoke()` uses. **Fixed** in this
branch's own final-review fix wave (not deferred to PR5b as the plan's decision record
originally intended) by giving `PolicyStore` a real `abatch` (delegates to
`asyncio.to_thread(self.batch, ...)`) — `docs/memory-policy.md` §10 is updated to match;
PR4's own put/aput policy injection (TTL/importance/cap enforcement) is unaffected since
those hooks run in `put`/`aput` themselves, not in `abatch`. Two dropped test cases (a
memory-tool end-to-end regression test and a missing-fields test) were backfilled in the
same fix wave — the missing regression test is exactly what let the bug ship past the
Task 8+9 review. 165 tests pass, pristine; `ruff check src tests scripts` clean.

**Process note:** mid-branch, execution switched to batched task dispatches + faster
review tiers (haiku) for the remaining tasks, at the user's request, on the
understanding that the mandatory final whole-branch review (full rigor, most capable
model, no discount) is the backstop. That is exactly what happened: the Critical finding
above surfaced there, not in the faster per-task loop — a real, observed tradeoff, not a
hypothetical one.

**Parked/deferred items carried forward (not fixed in PR5a, ledgered with reasoning
in the now-deleted SDD workspace, summarized here):**
- `agents/_react.py::_best_effort_tool_name` returns `"unknown_tool"` for any worker
  bound to 2+ tools (i.e. every worker except `benefit_check`) — **PR6 should fix this
  first**, before relying on tool-failure trace evidence for AC-12.
- `replan_count` is never incremented anywhere in PR5a — only the `supervisor_hops` cap
  is a live backstop against a routing loop right now. Correct scoping (PR6 owns
  `reflection.py`), but don't assume the replan cap is live before PR6 lands.
- Provider-namespace memory search in `intake` was deliberately not wired (only
  member-namespace is) — the provider NPI is only known after extraction, so a
  meaningful read needs a two-phase intake or a second tool call. PR5b/PR8 follow-up.
- `create_react_agent` is recompiled on every `run_worker_react` call (correctness is
  fine; revisit only if it shows up in a real latency trace).
- (from PR3, **still not done** — carry forward again) `medical_necessity`'s tool-call
  try/except must widen to `(RagIndexUnavailable, CorporaUnavailable)`, not just
  `RagIndexUnavailable`.

---

## PR5b — Graph Core, Part B  (merged `5b0068c`)

Closes **AC-01/02/03/04/05/10/11 in full, NFR-03 in full**. `agents/medical_necessity.py`
(MCP `criteria_check` + agentic RAG retrieval — the model decides when to call
`search_clinical_guidance` from its system prompt, not a code-level gate),
`agents/decision_draft.py` (synthesis + citation self-critique against
`retrieved_criteria`; writes a **critical**-importance memory record via a direct
`PolicyStore.put(...)` call on `disposition == "deny"` only, per design.md §6.3's literal
"denials/appeals = critical"), `agents/human_review.py` (`interrupt()` stub), `graph.py`
(the actual hand-rolled 7-node `StateGraph`), checkpointer wiring via
`AsyncSqliteSaver` — **the sync `SqliteSaver` design.md originally specified cannot run
any async operation at all** in this fully-async graph (verified empirically: raises
`NotImplementedError` on the first checkpoint read) — `docs/design.md` §1 corrected.
Pause/resume verified end-to-end across genuinely separate process invocations
(`scripts/run_pause_resume_test.py` → `traces/pause_resume_transcript.md`). Full-graph
evidence for AC-02/03 (`traces/run_full_case.json`, `route_clearcut.json`,
`route_ambiguous.json`) and NFR-03's canary (`traces/quarantine_canary.json`) all run
through the real compiled graph via `FakeToolCallingModel` (no `GEMINI_API_KEY` in this
environment). 196 tests pass, pristine; `ruff check src tests scripts` clean.

Key bugs caught only at review, invisible to every per-task check (same pattern as PR5a's
`PolicyStore.abatch` gap): (1) **task review** on the full-graph-evidence task caught all
three newly-committed trace files embedding real wall-clock timestamps, violating this
project's byte-stable-evidence convention — fixed by stripping `RouteStep.ts` at each
producer's serialization site (not centralized — a future producer can reintroduce this,
per the final review's own recommendation that PR7's `test_nfr04_trace_schema.py` should
assert repo-wide no committed trace contains a timestamp). (2) **final whole-branch
review** measured and reproduced two real crash bugs no per-task review could see, because
every committed run in the branch was scripted into the correct order: `settings
.recursion_limit` was never bound to the compiled graph (LangGraph's default 25-superstep
limit crashed a real run around `supervisor_hops == 8`, well before `config/routing.yaml`'s
`max_hops: 12` guardrail could ever fire — fixed via `compiled.with_config({"recursion_limit":
...})` in `make_graph`); the LLM router has no ordering guarantee and could pick
`medical_necessity`/`decision_draft` before their prerequisite state (`benefit`/
`necessity`) was set, causing an uncaught `AttributeError` — fixed with two new
deterministic `hard_route` guardrails (`benefit is None -> benefit_check`, `necessity is
None -> medical_necessity`) plus a defensive `needs_replan` fallback in both workers as a
second line of defense. Fixing the recursion limit also means `config/routing.yaml`'s hop
cap (12) is reachable in production for the first time — which unmasks the
`supervisor_hops`-never-resets-on-resume gap (below) as a live risk, not a dormant one.

**Ruling (recorded, not implemented):** design.md §3.3's documented "supervisor can
short-circuit not-covered cases directly to `decision_draft`, skipping
`medical_necessity`" optimization is currently unreachable (the new hard-route guardrail
forces `medical_necessity` first) — deliberately not re-implemented this PR; doing so
properly needs `decision_draft` to treat `necessity=None` as a valid input producing a
real (likely deny) decision, not just avoid a crash. Real design work for whoever next
touches supervisor routing, not a fix-wave item.

**Parked/deferred items carried forward (not fixed in PR5b):**
- **`supervisor_hops` never resets on resume** — `hard_route`'s hop-cap check only ever
  increments; nothing in application code (`human_review.py`/`supervisor.py`) resets it
  after a real resume. Task 6's own evidence script only works by manually passing
  `Command(update={"supervisor_hops": 0})` — no exposed application mechanism does this.
  **Must be fixed together with PR7's `pac resume`**, not independently — see above.
- **`agents/_react.py::_best_effort_tool_name`'s `"unknown_tool"` fallback is now
  universal** across every multi-tool worker in production config (confirmed: any worker
  bound to `load_pa_tools()`'s 3 real MCP tools always has `len(tools) > 1`). Promoted from
  "PR6 watch-out" to **required PR6 work** — NFR-07's tool-failure evidence is worthless
  without real per-tool attribution.
- **`decision_draft` only writes memory on `disposition == "deny"`** — approve/
  refer_clinical_review determinations are persisted nowhere, leaving the majority
  disposition path with no long-term memory trace (weakens AC-06/07's cross-session
  story). Named PR6/PR8 follow-up: a `routine`-importance write on every disposition.
- **`state["messages"]` is still never written by any worker** — `summarize`'s
  `SummarizationNode` (NFR-08) runs on an empty thread on every real turn; the mechanism
  and its PR5a evidence are genuine, but it's inert in the actual system. Real design
  decision (worker-authored messages could affect prompt construction elsewhere), not a
  quick fix — still open, not yet assigned to a PR.
- (from PR3, **now done**) the `(RagIndexUnavailable, CorporaUnavailable)` except-clause
  widening — PR5b Task 1 fixed it at the source, in `rag/tool.py` itself, rather than in
  the worker. No longer a forward watch-out.
- `mcp_tools` is currently the full unfiltered list shared across
  `intake`/`benefit_check`/`medical_necessity` (each worker's model technically has every
  MCP tool bound, constrained only by its own system prompt's silence on the others) —
  accepted as a first approach; revisit if a test ever shows a worker mis-calling an
  out-of-scope tool.
- AC-05's committed evidence state is hand-seeded (the hop-cap trick), not reached via a
  case that organically hits `human_review` — Task 7 later produced exactly such a case
  (the ambiguous run); rebuilding AC-05's evidence on it would be a strictly stronger
  proof and would retire the `supervisor_hops` reset hack from the evidence path.
  Opportunistic, not urgent.

---

## PR6 — Reflection & Self-Healing  (merged `c5f9677`)

Closes **AC-12, NFR-07** in full. `reflection.py` — two mechanisms: (1)
`attribute_tool_errors(tools)` wraps each tool's `.func`/`.coroutine` in place
(idempotent via a `_pa_attributed` marker — safe across the 3 worker builders that
share `mcp_tools`) so a tool failure is tagged with its REAL name via a new
`AttributedToolError`, closing the `"unknown_tool"` gap `_best_effort_tool_name` had
for every worker bound to 2+ tools (PR5a/5b's carried-forward item); (2)
`run_worker_react_resilient(...)` wraps one whole worker turn in `asyncio.wait_for`
(`worker_timeout_seconds`), `tenacity` exponential-backoff retries
(`max_tool_retries`) on a transient `WorkerToolError`, and exactly one separate
fallback attempt on a lite model (`model_agent_lite`) when the structured-output call
itself fails validation (`WorkerOutputError`) — wired into all four ReAct workers.
`supervisor.py` — `hard_route`'s cap-exhaustion check broadened to
`(tool_failures or needs_replan) and replan_count >= max_replans`; `build_supervisor_
node` now consumes a pending `needs_replan` once per turn (increments `replan_count`,
builds an `effective_state` visible to `hard_route`/`route_with_llm` on the SAME
turn, clears `needs_replan`); `route_with_llm` gained a reflection hint in its
prompt. `medical_necessity.py` sets `needs_replan=True` on `confidence < tau` or
`criteria_status="indeterminate"` WITHOUT withholding `necessity`/`retrieved_
criteria` — the low-confidence "loop back" is a real, reachable LLM-router choice
(design.md's "supervisor loops back with a hint"), not a new deterministic
`hard_route` rule; this was a deliberate design ruling, confirmed independently
correct by the final whole-branch review, with one disclosed cost: the trigger is
proven *reachable*, not proven to *occur* — a real Gemini call may pick
`human_review` directly instead of looping back, same as `tests/_full_case.py`'s
already-`done` AC-03 ambiguous case does. `human_review.py` now resets
`supervisor_hops`/`replan_count` to 0 on resume (a human just intervened — fresh
attempt budget), closing PR5b's carried-forward gap; testable in PR6 without PR7's
`pac resume` existing (`InMemorySaver` + `Command(resume=...)`, no manual
`update=` needed). AC-12 evidence: `tests/_reflection_case.py` (4 scripted scenarios
— tool-failure recover/exhaust, low-confidence recover/exhaust — through the REAL
compiled graph), `tests/test_ac12_reflection.py`, `scripts/reflection_demo.py` →
`traces/reflection_tool_failure.json` / `traces/reflection_low_confidence.json`
(byte-stable). NFR-07: `tests/test_nfr07_degradation.py` (persistent tool failure +
persistent timeout, both end cleanly at `human_review`, never an unhandled
exception). 227 tests pass (222 fast + 5 slow-skipped without a key), pristine;
`ruff check src tests scripts` clean.

Key bugs caught and fixed, several only at the final whole-branch review (same
pattern as every prior PR — see PR4/PR5a/PR5b's own entries): (1) **Task 1 fix
round**: a test mutated the real `search_clinical_guidance` module-level singleton's
new `_pa_attributed` idempotency marker with no teardown, permanently corrupting it
for the rest of the pytest process — fixed via `.model_copy()` on a throwaway
instance instead. (2) **Task 2**: `run_worker_react_resilient`'s tenacity retries
exhaust a single-shot `FakeToolCallingModel` script on the 2nd attempt, masking
correct attribution as `"unknown_tool"` again — fixed by pinning `max_tool_retries=1`
in the three affected worker tests (they test the single-failure contract, not
retry behavior). (3) **Task 4**: found (while implementing, not via review) that
`StateGraph(dict)` combined with a `PACaseState`-annotated node produces incorrect
merge behavior on resume — a real LangGraph introspection quirk, worked around by
using `StateGraph(PACaseState)` in the test (which also matches production); also
found a genuine regression Task 4's own fix caused against a *pre-existing* test
that had pinned the OLD buggy "resume never resets hops" behavior — updated+renamed
rather than left stale. (4) **Final whole-branch review** (opus, independently
re-ran the fast suite + ruff + hand-traced the full exception chain end to end):
`decision_draft.py` was the only one of the four workers not catching
`WorkerToolError`, so PR6's own new per-turn timeout could escape `graph.ainvoke()`
completely uncaught — in the same branch that marks NFR-07 `done` claiming "never an
unhandled exception." Fixed by adding the same handler the other three workers
already had. Also: a timeout-sourced `WorkerToolError` was being retried by tenacity
like a real transient error, pushing worst-case per-node-visit latency to
`max_tool_retries × worker_timeout_seconds` (~90s) — fixed via a new
`WorkerTimeoutError(WorkerToolError)` subclass excluded from the retry predicate (a
blown time budget isn't "transient"), still caught unchanged by every worker's
existing `except WorkerToolError`. One scoped re-review confirmed both fixes clean.

**Process note:** partway through, the user asked to skip per-task review to
conserve API budget — Tasks 3(fix)/4/5/6 were implemented directly by the session
(no subagent dispatch, no per-task reviewer) and self-verified via targeted test
runs; the mandatory final whole-branch review (opus, full rigor) was kept as the
sole remaining gate and is exactly what caught the two Important findings above —
the same "faster loop, rigorous backstop" tradeoff PR5a's own process note
described, observed again. Separately, Task 2's subagent implementer stalled twice
(ended its turn mid-task with no commit; then ran 32 minutes with no output after
being resumed) — the session took over directly rather than re-dispatching a third
time, keeping the subagent's correct edits (verified via diff read) and fixing a
real bug it hadn't reached yet.

**Parked/deferred items carried forward (not fixed in PR6, ledgered with reasoning
in `.superpowers/sdd/implementation-plan-pr6/progress.md`, now deleted per the SDD
skill's finish step — summarized here):**
- `ToolFailure.attempt` is hardcoded to `1` everywhere it's constructed, so it never
  reflects how many tenacity attempts a failure actually burned — the committed
  `traces/reflection_tool_failure.json` trace under-reports this (only the fake
  tool's own error-message string happens to reveal the real count). `reflection.py`
  has the real `retry_state.attempt_number` available; stamping it onto the
  re-raised `WorkerToolError` would fix this.
- `supervisor.py`'s `route_with_llm` reflection hint only covers 2 of the 3
  `needs_replan` triggers — `decision_draft`'s self-critique failure produces an
  empty hint. Also a latent (currently unreachable) `AttributeError` trip-wire:
  `necessity is not None` doesn't guard against a dict placeholder lacking
  `.confidence`.
- `attribute_tool_errors`' wrapper closures erase the wrapped callable's signature
  (no `functools.wraps`) — verified safe today (no tool in this codebase resolves a
  `config: RunnableConfig` parameter via introspection; langmem's namespace
  templating uses a contextvar instead), but a future tool that does would lose
  config injection silently.
- Inconsistent `get_settings.cache_clear()` symmetry across worker tests that
  override env-based settings after a fixture (like `memory_store`) has already
  primed the `lru_cache` — `test_agent_benefit_check.py`'s equivalent test currently
  passes for the "wrong reason" (single-tool worker, so `_best_effort_tool_name`
  would have guessed right anyway).
- `tool_failures`' `operator.add` reducer never clears, so once any failure occurs
  in a case, the broadened cap check stays permanently armed — combined with the
  cap sitting above the FINISH rule in `hard_route`, a case that fully recovers
  from `replan_count == max_replans` worth of failures routes to `human_review`
  instead of `FINISH` even though `decision` is set. Judged intentional-as-an-
  explicit-exit-condition by the final review, but untested and undocumented as
  such.
- `test_medical_necessity_calls_rag_when_indeterminate` (pre-existing, not touched
  by PR6) is unusually slow (~20-40s, real `bge-small` embedding model load) and
  flaked with a real `KeyError` failure once during this PR's own merge-verification
  step, passing cleanly on immediate retry — not reproduced a second time, and the
  final review had already independently flagged this exact test as an outlier
  before the flake occurred. Plausibly HF Hub resolution variability, but PR6's new
  `worker_timeout_seconds` default (30s) is now a real hard ceiling this test sits
  close to, where none existed before — worth switching this test to `FakeEmbedder`
  (like every other RAG test already does) rather than hoping the real model keeps
  loading fast enough, independent of anything else in PR6.
- 2 gaps in the plan itself, not the implementation: the plan never reasoned about
  whether a timeout should be retryable before specifying both "timeout becomes a
  WorkerToolError" and "retry on WorkerToolError" as separate facts; the plan's own
  "decision_draft's except clause is out of scope for this task" ruling created the
  scope seam that the NFR-07-done claim then fell into.

---

## PR7 — Interfaces  (merged `af75f5e`)

Closes **AC-10 (full), NFR-01, NFR-02, NFR-04, NFR-06**. `cli.py` — Typer app `pac`
with `ingest`, `submit`, `resume`, `memory`, `persistence-test`, `compare`, `demo`,
`all`; `_open_state`/`_open_graph`/`_open_mcp_tools` async-context-manager helpers
keep one live MCP stdio session + checkpointer/store open for a whole graph
invocation; `_thread_status(graph, thread)` classifies a case_id's thread as
new/paused/finished via `graph.aget_state(thread).next`/`.created_at` (empirically
verified against all three states) so `submit`/`compare` can refuse to
resubmit/re-run with the RIGHT advice (`pac resume` only when genuinely paused).
`app/streamlit_app.py` — routing trail, artifact viewer, memory panel, built from
pure `_route_rows`/`_artifact_rows`/`_memory_panel_rows` functions sharing
`cli.py`'s `_open_graph`/`_open_state`. `single_agent.py` — `run_single_agent` binds
the same MCP tools + `search_clinical_guidance`/memory tools onto one
`run_worker_react_resilient` call (no supervisor/routing), wired into `pac compare`
alongside the multi-agent graph over one shared MCP session. `scripts/
mcp_transcript_demo.py` regenerates a genuine (not hand-written) MCP tool-call
transcript from a real adapter session → `traces/mcp_toolcall_transcript.md`/
`mcp_tool_calls.jsonl`. `docs/single-vs-multi-agent.md`, `docs/agent-patterns.md`
new; `docs/rubric-coverage.md` rewritten as a real 22-parameter table (AC-10 row
corrected to not overstate: a full compiled-graph run through both a real MCP
session AND a real LLM together stays deferred — no `GEMINI_API_KEY` in this
environment). `.github/workflows/tests.yml` — ruff + fast pytest on push/PR,
Python 3.11. 268 tests pass, ruff clean.

**Process note:** per-task review was dropped for this PR (budget), keeping only
one final whole-branch review before merge — a deliberate scope reduction, agreed
with the user in advance. This does not weaken any AC/NFR: the rubric scores
committed evidence (tests, traces, docs), not review cadence, and the merge gate
was still "no unresolved Critical/Important finding," same as every prior PR.

Key bugs caught and fixed, all at the final whole-branch review (no per-task
review this time, so everything surfaced at once — see the process note above):
**Critical** — `_build_graph_for_case` returned the compiled graph from *inside*
`async with pa_session(...)`, so the MCP subprocess was already closed before any
caller could use the graph; every real (non-mocked) CLI invocation would break on
its first live tool call. Fixed by converting it to an `@asynccontextmanager`
(`_open_graph`) and updating every call site to `async with _open_graph(...) as
graph`. Plus 9 Important/Moderate findings in the same pass: missing traces on
resume/exception, `pac all` aborting the whole run on one failed `compare`, no real
embedder ever passed anywhere (semantic memory silently disabled in every real
run), `compare()` not handling `__interrupt__`, non-idempotent `case_id`-as-
`thread_id` reuse, `single_agent.py`'s too-low `recursion_limit`, 4x duplicated
resource-lifecycle scaffolding, and `rubric-coverage.md` overstating AC-10 — all
fixed in one fix-wave commit (`165553f`).

A scoped re-review of that fix wave (checking only the diff it introduced, not the
whole branch again) found 3 more issues, one of them a design bug and not just a
missed edge case: the idempotency guard used mere checkpoint *existence*
(`aget_tuple`), which can't distinguish paused from finished — a genuinely
completed case_id got told to `pac resume` (wrong advice, and the CLI's own
`_thread_status` docstring/tests now assert the opposite), and `typer.Exit` was
being swallowed by `run_all()`'s bare `except Exception`, breaking `pac all`'s
re-runnability. `compare()` had no equivalent guard at all. `_real_embedder`'s
docstring claimed laziness/cheapness that wasn't true in practice (`memory_store.
setup()` eagerly calls `embed_query`). Fixed with the `aget_state`-based
`_thread_status` helper described above, `compare()` gaining the same guard, and a
corrected docstring — commit `80f38d1`. This second fix wave was NOT sent for a
further re-review pass (user call, given its narrow scope and its own dedicated
new test coverage) before merging.

---

## NEXT: PR8 — Good-to-Haves  (branch TBD, off `main` @ `af75f5e`)

**Scope (docs/rubric-coverage.md's "Good-to-Haves" line, not separately scored):**
a 2nd MCP server, a criteria-met fast-path, an importance-weighted background
memory manager, Streamlit memory-panel enrichment, and surfacing `pac compare`'s
single-vs-multi distinction more directly in the UI. All 22 rubric parameters (100
marks) are already `Done` as of PR7 — PR8 is polish, not required for the graded
submission.

**Resume:** `cd D:/Aru/NYU/Virtusa/prior-auth-copilot`, confirm `git log
--first-parent` shows `Merge PR7`, then decide with the user which Good-to-Haves
(if any) are worth the remaining time before treating this project as feature-complete.

**Operational lessons from PR7, worth carrying forward:**
- Dropping per-task review (budget-driven) meant all 10 fix-wave findings from
  PR7's Task work surfaced in one place at the final review, followed by a second
  round of 3 more in the fix wave itself — a scoped re-review of a fix wave is
  worth doing by default when the fix wave itself was non-trivial (touched control
  flow, not just a docstring or a single guard clause), even when per-task review
  was otherwise skipped.
- `aget_state(thread).next`/`.created_at` is the reliable way to distinguish
  new/paused/finished LangGraph threads — verify empirically (a throwaway script
  against all three states) rather than trusting checkpoint *existence* alone,
  which conflates paused and finished.
- When a stray/duplicate agent is caught mid-task (two writers touching the same
  file), kill the redundant one immediately in that same turn rather than letting
  it keep running in the background — see the `feedback_avoid-redundant-agents`
  memory.
