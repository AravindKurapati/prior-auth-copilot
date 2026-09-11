# Memory Policy — Prior-Authorization Copilot

Memory Systems category — AC-06 (tiered recall), AC-07 (cross-session persistence),
AC-08 (eviction/importance policy). Design context: `docs/design.md` §6. Implementation:
`src/pa_copilot/memory/{policy,store,embeddings,tools,working}.py`, config
`config/memory.yaml` / `pa_copilot.config.MemoryConfig`.

---

## 1. Tiers

| Tier | Store | Scope | Module | Persisted by |
|---|---|---|---|---|
| **1 — working** | `state["working_memory"]["facts"]` | one case / thread | `memory/working.py`: `remember`, `recall`, `search_working` — pure dict transforms, no store/model dependency | `SqliteSaver` checkpointer (wired in PR5); this PR exercises the dict ops directly |
| **2 — long-term / semantic** | `PolicyStore` (`SqliteStore` subclass) over `.pa_memory.db`, optional `sqlite-vec` index | cross-thread, cross-session | `memory/store.py`, `memory/policy.py`, `memory/embeddings.py` | on-disk SQLite; survives process exit |

Tier-2 namespaces (`config/memory.yaml`): `("pa","member",<id>)`, `("pa","provider",<npi>)`,
`("pa","policy_notes")`, `("pa","episodic")`. Agent-managed via
`memory/tools.py::build_memory_tools` — thin wrappers over langmem's
`create_manage_memory_tool` / `create_search_memory_tool` bound to an explicit `store=` (the
langmem default of resolving the store from graph config is not used here, so tests and CLI
scripts can pass an arbitrary `PolicyStore` instance directly).

---

## 2. TTL

`langgraph.store.base.TTLConfig` only carries a single **global** `default_ttl` — one number
for the whole store, not one per namespace. That's too coarse: `episodic` should expire in
90 days, `member`/`provider` in 365, and `policy_notes` never.

`PolicyStore` works around this by computing a **per-item** TTL on every write instead of
relying on the store-wide default:

- `PolicyStore.put` / `PolicyStore.aput` override the base `SqliteStore` methods. Before
  delegating to `super().put(...)`, `_prep()` calls `policy.ttl_minutes_for(namespace, mem)`
  whenever the caller didn't already pass an explicit `ttl` (checked via the `NOT_PROVIDED`
  sentinel), converts the namespace's configured `ttl_days` to minutes, and passes that as
  the per-call `ttl` argument that `SqliteStore.put` already accepts natively.
  `ttl_minutes_for` returns `None` for `policy_notes` (`ttl_days: null`), so those rows never
  expire.
- The store is still constructed with `TTLConfig(refresh_on_read=True)` — that global config
  only turns on the refresh-on-read *behavior*; the actual expiry horizon for each row comes
  from the per-item `ttl` `PolicyStore` injects, not from `TTLConfig.default_ttl` (left unset).
- `refresh_on_read=True` means a normal read (`store.get`/`store.search` with default
  `refresh_ttl`) resets that item's expiry clock — a member's record that keeps getting
  looked up stays alive.
- `policy.sweep_expired(store)` — a thin wrapper over `store.sweep_ttl()` — does the actual
  deletion of rows past `expires_at`. It's opportunistic: called on startup / by the `pac`
  CLI (PR7) and by `store.start_ttl_sweeper()` once the graph runs long-lived (PR5). Nothing
  in PR4 auto-sweeps on a timer.

## 3. Importance weighting

Every value dict written through `PolicyStore` gets an `importance` field if the caller
didn't supply one — `_prep()` calls `policy.importance_of(value, mem)`, which falls back to
`MemoryConfig.default_importance` (`"routine"`, from `config/memory.yaml`) when the field is
missing or holds a label the config doesn't recognize.

Weights (`config/memory.yaml: importance_weights`): `{routine: 1.0, notable: 1.5, critical:
3.0}`. Denials and appeals are written with `importance: "critical"` by the caller (e.g.
`decision_draft` in PR5) — `policy.importance_weight()` looks the label up in this table.

## 4. Ranked search

`policy.search_ranked(store, namespace, query, *, limit=None, mem=None, now=None, pool=50)`
pulls a generous pool from `store.search(...)` (semantic when the sqlite-vec index is up,
recency/keyword-only otherwise — `use_query` is dropped to `None` when
`store.semantic_index_available` is `False`, so a dead index degrades gracefully instead of
erroring), then re-sorts that pool in Python by `policy.rank_key`:

```
rank_key = semantic_score * importance_weight(value) * recency_decay(updated_at, half_life_days)
```

`recency_decay` is a half-life exponential (`config/memory.yaml: recency_half_life_days: 30`)
— `0.5 ** (age_days / half_life_days)`, `1.0` for anything read as "now or future" or
whose timestamp can't be parsed. `semantic_score` defaults to `1.0` when the store didn't
return one (no-index mode), so ranking there is pure importance × recency.

## 5. LRU cap (soft against `critical`)

Each namespace has a `cap` (`config/memory.yaml`). `PolicyStore.put`/`aput` call
`policy.enforce_cap(self, namespace, mem=self._mem)` after every successful write.
`enforce_cap`:

1. Lists everything currently in the namespace (`store.search(namespace, limit=10_000,
   refresh_ttl=False)` — see the note below on why `refresh_ttl=False` matters here).
2. If the namespace isn't over cap, returns `[]`.
3. Otherwise sorts ascending by `rank_key` (worst first) and deletes items one at a time
   until back at cap, **skipping any item whose `importance == "critical"`**.

The cap is intentionally **soft against `critical`**: if a namespace holds more `critical`
items than its cap, they are left in place rather than evicted — losing a denial/appeal
record to make room is worse than a namespace running slightly over its configured size.
This is a documented limitation, not a bug: a namespace flooded with `critical` writes has
no automatic ceiling.

**Fixed during review:** `enforce_cap`'s internal listing scan passes `refresh_ttl=False`.
Without that, every single write's cap-check would call `store.search()` over the whole
namespace with the default `refresh_ttl=True`, silently refreshing the TTL clock on every
*other* item in the namespace on every write — turning "N days since this item was last
touched" into "N days since anything in the namespace was written," which defeats
per-namespace expiry entirely. `enforce_cap`'s own listing is bookkeeping, not a real read of
any one item, so it must not count as a "read" for TTL-refresh purposes.

## 6. sqlite-vec fallback

`open_memory_store()` tries to build a `PolicyStore` with a semantic index (`sqlite-vec` +
local bge-small embeddings via `memory/embeddings.py::as_embeddings`/`embedding_dims`,
wrapping `rag.embedder.BgeEmbedder`). If `store.setup()` raises while loading the `sqlite-vec`
extension, `open_memory_store` closes that connection, rebuilds the store with `index=None`,
records the original exception as `store.semantic_error` (a `repr(exc)` string, not raised),
and calls `setup()` again. If the no-index rebuild *also* fails, the connection is closed and
the exception propagates — there's no silent third fallback.

With `index=None`, `store.semantic_index_available` is `False`; `search_ranked` drops the
semantic query and ranks purely by importance × recency. AC-06/07/08 all still hold in this
mode — semantic *recall quality* degrades (no embedding similarity), but TTL, importance,
cap eviction, and cross-session persistence are unaffected, since none of those depend on
the vector index.

**Also fixed during review:** `open_memory_store` closes the raw sqlite3 connection on
*both* failure edges (index-setup failure and no-index-rebuild failure) before propagating
or retrying — the original code leaked the connection object on Windows, where an unclosed
sqlite3 handle can leave the file locked for a subsequent open in the same test process.

## 7. Cross-session persistence (AC-07)

"Session" = a fresh set of in-memory objects (store, and later graph/checkpointer) pointing
at the same on-disk SQLite file. Two forms of evidence:

- **In-process** — `tests/test_memory_persistence.py`: Session A writes a record into
  `("pa","member",<id>)`, all in-memory objects are torn down, Session B opens a **new**
  `PolicyStore` via `open_memory_store()` against the same path and asserts (exact key/value
  match, no LLM judgment) the record is still there and its TTL/importance stamping
  survived the round trip.
- **Cross-process** — `scripts/run_persistence_test.py` spawns two real, separate `python`
  subprocesses against the same `.pa_memory.db` file: the first writes, the second reads and
  asserts. Combined, masked-timestamp/pid stdout is committed as `traces/memory_persistence.log`
  (byte-stable — no raw timestamps or pids). The script surfaces a crashing child's stderr to
  its own stderr so a real failure is visible in CI/local logs without polluting the
  byte-stable committed log itself.

## 8. Testing gotcha: SQLite timestamp granularity

`created_at`/`updated_at` in the underlying `store` table are stamped via SQL
`CURRENT_TIMESTAMP`, which has **1-second granularity**. Items written back-to-back inside a
single test tie on `recency_decay` (both round to `~1.0`), so a naive eviction test can pass
for the wrong reason — Python's stable sort falls back to insertion order on a tie, which
looks like "recency worked" even if `recency_decay` were a no-op.

`tests/test_ac08_eviction.py::test_overflow_evicts_lowest_non_critical` avoids this by
directly backdating the item that should be evicted with a raw SQL `UPDATE ... SET
created_at = ?, updated_at = ?` to a clearly-older timestamp (`2000-01-01`), and deliberately
writing that item *third* (not first) so that if recency ranking were broken and eviction
fell back to insertion order, a *different* item would be evicted instead — making the test
fail loudly rather than pass by coincidence. Anyone writing a similar recency-eviction test
against a SQLite-backed store should use the same raw-SQL-backdating technique rather than
relying on write-order timing gaps.

---

## 9. Good-to-Have (PR8)

`langmem.create_memory_store_manager` — a background process that extracts and writes
importance-weighted memories from conversation history automatically, instead of relying on
the agent explicitly calling `manage_memory`. Not implemented in PR4; noted here as the
natural next step for the memory subsystem.

---

## 10. Known limitation carried into PR5 (not fixed here)

LangMem's memory tools (`build_memory_tools`) only work through **synchronous** `.invoke()`
against `PolicyStore`. Calling `.ainvoke()` raises `NotImplementedError`, because
`SqliteStore.abatch` — the langgraph library's own override, not first-party code in this
repo — unconditionally raises for async batching. `PolicyStore.aput` is written and does
compile, but nothing in PR4 proves it works end-to-end through an async caller, because
LangMem's async tool path can never reach it (LangMem batches through `store.abatch`, not
`store.aput`, when invoked async).

PR5's workers are async ReAct loops. If any worker calls these memory tools via `ainvoke`,
it will hit `NotImplementedError`. This is an open design question for PR5 — call the tools
synchronously from within the async worker, wrap them, or give `PolicyStore` a real
`abatch` override — not a solved problem in this PR.
