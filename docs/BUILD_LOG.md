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

## NEXT: PR4 — Memory Subsystem  (branch `feat/memory`, off `main` @ `2d8762a`)

**Scope (design.md §6; AC-06, AC-07, AC-08):**
- Tiered memory: short-term working (`state["working_memory"]` + `messages`, will be
  persisted by `SqliteSaver` — wired in PR5) + long-term semantic (`SqliteStore` +
  `sqlite-vec`, local bge-small embeddings). Namespaces `("pa","member",<id>)`,
  `("pa","provider",<npi>)`, `("pa","policy_notes")`, `("pa","episodic")`.
- LangMem `create_manage_memory_tool` / `create_search_memory_tool` over the store.
- `memory/policy.py` — eviction/importance: native `SqliteStore(ttl=TTLConfig(...))` +
  importance weighting (`routine`/`notable`/`critical`, denials/appeals = critical) +
  per-namespace LRU cap. `search_ranked()` = `semantic × importance × recency`.
- **AC-07 deterministic cross-session test** (`tests/test_memory_persistence.py` +
  `scripts/run_persistence_test.py` — two real `python` processes, same on-disk store) →
  committed `traces/memory_persistence.log`.
- `docs/memory-policy.md`.

**PR4 watch-outs:**
- `Settings.memory` is already a frozen `MemoryConfig` (PR1) — use it; add an autouse
  `os.environ` snapshot/restore fixture to `tests/conftest.py` (PR2 parked this — a
  `GOOGLE_API_KEY` raw-`os.environ` write in `test_env_overrides_and_key_read` currently
  leaks; PR4's env-heavy tests need the snapshot fixture).
- `SqliteStore` semantic index needs an embedder — reuse `rag.embedder.BgeEmbedder` (or a
  LangChain `Embeddings` wrapper around it). Fast tests inject a fake (`tests/_fakes.py`).
- `sqlite-vec` may fail to load on some machines — `memory/store.py` should fall back to a
  no-index `SqliteStore` (keyword search only) and still pass AC-06/07/08.
- Real bge-small `@slow`; deterministic tests use the fake.
- Keep `.pa_memory.db*` gitignored (already is).

**Resume:** `cd D:/Aru/NYU/Virtusa/prior-auth-copilot`, confirm `git log --first-parent`
shows `Merge PR3`, then write `docs/implementation-plan-pr4.md` and run the SDD cycle.
Prior rulings R1–R29 are in `.superpowers/sdd/implementation-plan-pr{1,2,3}/progress.md`.
