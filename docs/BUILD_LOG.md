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
- Local `--no-ff` merges to `main` are pre-authorized for this workflow. **Not pushed** —
  no GitHub remote yet (deferred; user's choice was "new public GitHub repo, GitLab later").
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

---

## NEXT: PR5 — The Graph  (branch `feat/graph-core`, off `main` @ `050b488`)

**Scope (design.md §3; AC-02, AC-03, AC-05; NFR-03, NFR-08):**
- `supervisor.py` — deterministic guardrails + LLM router (`RouterDecision`) over
  `state["next"]`.
- `agents/` — `intake`, `benefit_check`, `medical_necessity`, `decision_draft`,
  `human_review`, each a small internal ReAct loop emitting one validated Pydantic object.
- `graph.py` — the hand-rolled `StateGraph` topology (`summarize` → `supervisor` →
  conditional → workers → `summarize` → `supervisor` ... → `FINISH`/`human_review`
  `interrupt()`), async `make_graph()`.
- `context/` — `quarantine.py` (NFR-03: untrusted `raw_provider_text` isolation),
  `summarization.py` (`SummarizationNode`, NFR-08), `assembly.py` (write/select helpers).
- Checkpointer wiring (AC-05): `graph.compile(checkpointer=SqliteSaver..., store=PolicyStore...)`;
  `pac submit` / `pac resume` as two separate process invocations,
  `traces/pause_resume_transcript.md`.

**Parked items carried forward:**
- (from PR3) `tool._service_name` → `data_access.list_policies()` → `load_corpora()` can
  raise `CorporaUnavailable`, which the RAG tool's own `except RagIndexUnavailable` does not
  catch. `medical_necessity`'s tool-call try/except should widen to
  `(RagIndexUnavailable, CorporaUnavailable)` → `[]`.
- (from PR2, **now done**) the autouse `os.environ` snapshot/restore fixture — landed in PR4
  as `tests/conftest.py::_env_snapshot`. No longer a forward watch-out.
- **New — async LangMem tools vs. async workers (open design question).** `PolicyStore.aput`
  cannot succeed from any async caller, period: `BaseStore.aput` unconditionally calls
  `await self.abatch(...)`, and `SqliteStore.abatch` unconditionally raises
  `NotImplementedError` for any async batch operation — this is true whether the caller is
  LangMem's `.ainvoke()` or a direct first-party `await store.aput(...)`. PR5's workers are
  async ReAct loops. Before wiring `build_memory_tools()` output into any worker, decide: call
  the memory tools synchronously from inside the async worker, wrap them in a sync-to-async
  adapter, or (preferred) give `PolicyStore` a real `abatch` override so `aput` has a working
  path at all. Not solved in PR4 — see `docs/memory-policy.md` §10.
- **New — `manage_memory` tool writes are always `routine` (open design question).**
  LangMem's `create_manage_memory_tool` hardcodes `value={"content": ...}` at the top level of
  what it passes to `store.put` — there is no way for an agent using that tool to set
  `importance`, so every write made through it is permanently stamped `routine`. A worker that
  needs to write a `critical` record (e.g. `decision_draft` recording a denial/appeal) must
  call `PolicyStore.put(...)` directly with `value={"importance": "critical", ...}`, or PR5
  needs its own first-party wrapper tool exposing an `importance` parameter — not the generic
  `manage_memory` tool. See `docs/memory-policy.md` §§1, 3.

**Resume:** `cd D:/Aru/NYU/Virtusa/prior-auth-copilot`, confirm `git log --first-parent`
shows `Merge PR4`, then write `docs/implementation-plan-pr5.md` and run the SDD cycle.
Prior rulings are in `.superpowers/sdd/implementation-plan-pr{1,2,3,4}/progress.md`.
