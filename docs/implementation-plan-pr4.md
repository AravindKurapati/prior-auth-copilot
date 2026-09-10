# PR4 — Memory Subsystem: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tiered memory subsystem — short-term working memory helpers over
`state["working_memory"]`, plus a long-term semantic store (`SqliteStore` + `sqlite-vec`,
local bge-small embeddings) governed by a per-namespace TTL / importance / LRU-cap policy,
with LangMem `manage_memory` / `search_memory` tools bound to it. Closes AC-06, AC-07,
AC-08.

**Architecture:** `memory/embeddings.py` adapts any duck-typed embedder (real `BgeEmbedder`
or a test fake) to the LangChain `Embeddings` interface the store wants. `memory/policy.py`
is pure policy math — namespace → TTL minutes, importance weight, recency decay, combined
rank key, cap enforcement, ranked search — no library coupling beyond a store handle passed
in. `memory/store.py` defines `PolicyStore(SqliteStore)`, which intercepts every `put` /
`aput` (LangMem tool writes included) to inject the namespace's TTL and a default
importance, then enforces the LRU cap; its `open_memory_store(...)` factory builds the
sqlite-vec-indexed store and **falls back to a non-semantic store** if the extension will
not load. `memory/tools.py` wraps the two LangMem tools. `memory/working.py` is the Tier-1
scratch API. Two committed traces (`tiered_memory_recall.json`, `memory_persistence.log`)
each get an in-repo producer script and a byte-stability guard test.

**Tech Stack:** `langgraph-checkpoint-sqlite` 3.0.3 (`langgraph.store.sqlite.SqliteStore`),
`langgraph.store.base` (`TTLConfig`, `SearchItem`, `NOT_PROVIDED`), `langmem` 0.0.30
(`create_manage_memory_tool` / `create_search_memory_tool`), `sqlite-vec` 0.1.9,
`sentence-transformers` (`BAAI/bge-small-en-v1.5`, CPU, reused from `rag/`),
`langchain-core` `Embeddings`, Pydantic-free plain dataclasses for config, `pytest`.

**Spec:** `docs/design.md` §6 (authoritative), plus §5.1 (Write strategy),
`specs/acceptance-criteria.md` AC-06/07/08. Design decision approved 2026-09-10:
per-namespace policy is enforced by a `PolicyStore` subclass (Approach A), not an
out-of-store helper.

## Global Constraints

- Python `>=3.11`. Package `pa_copilot` under `src/`. No Docker, no DB service — the store
  is a local SQLite file (`./.pa_memory.db`, gitignored as `.pa_memory.db*` and `*.sqlite3`).
- Stack stays **langchain 0.3.x / langchain-core 0.3.x** — do NOT upgrade. Pins already in
  `pyproject.toml`: `langgraph>=1.0,<2`, `langgraph-checkpoint-sqlite>=3,<4`, `langmem>=0.0.20`.
- **No `GEMINI_API_KEY` / `GOOGLE_API_KEY` in this environment.** Nothing in PR4 calls an
  LLM. Every test is deterministic. Embedding uses **local** models only.
- The **real bge-small model DOES download** here (confirmed in PR3), so a `@pytest.mark.slow`
  real-embedder test is allowed — but the CI gate and every AC test run on the deterministic
  `tests/_fakes.py::FakeEmbedder` (crc32 bag-of-words, `DIM = 64`).
- Test output must be **pristine** under `pyproject` `filterwarnings = ["error"]` (+ the two
  existing narrow `chromadb` ignores). Any new warning fails the suite. The `langgraph`
  import-time `LangChainPendingDeprecationWarning` is already contained by
  `tests/conftest.py`'s existing filter setup — do not re-suppress it.
- Committed evidence under `traces/` is **byte-stable** across regeneration: no timestamps
  in the committed bytes (mask volatile fields as `"<ts>"`); every file writer uses
  `write_text(..., encoding="utf-8", newline="\n")` and `json.dumps(..., indent=2) + "\n"`
  (mirror `scripts/ingest_rag.py`).
- Every AC test + artifact carries its `AC-06` / `AC-07` / `AC-08` id in a name, docstring,
  or payload key. `tests/test_ac_traceability.py` fails a `done`/`partial` ledger row that
  names a nonexistent test file — the filenames in this plan are the contract.
- Windows: the store owns a live `sqlite3` connection — never `shutil.rmtree` a db file
  whose connection is open; close the store (`store.conn.close()`) first. Tests use
  `tmp_path` (pytest owns cleanup) or the `memory_store` fixture (closes on teardown).
- Work on branch `feat/memory` off `main`. The controller does the `--no-ff` merge after a
  whole-branch review — do NOT merge in a task.
- TDD: failing test first, run it red, minimal impl, run it green, commit. Frequent commits.

## Interfaces from PR1–PR3 (on `main` @ `3de3eb5`)

- `pa_copilot.config.load_settings(config_dir="config", env_file=".env") -> Settings` and
  `get_settings()` (`@functools.lru_cache(maxsize=1)`, autouse-cleared each test by
  `tests/conftest.py::_clear_settings_cache`). `Settings` is a `@dataclass(frozen=True)`.
  Relevant fields: `.memory_db` (`"./.pa_memory.db"`), `.embedding_model`
  (`"BAAI/bge-small-en-v1.5"`), `.rag_query_prefix` (`""`), `.traces_dir`, `.repo_root`,
  `.memory` (a `MemoryConfig`). **This PR adds two `MemoryConfig` fields — see Task 2.**
- `pa_copilot.config.MemoryConfig` — `@dataclass(frozen=True)`:
  `namespaces: dict[str, NamespacePolicy]`, `importance_weights: dict[str, float]`,
  `recency_half_life_days: int`, `raw: dict`.
- `pa_copilot.config.NamespacePolicy` — `@dataclass(frozen=True)`:
  `ttl_days: int | None`, `cap: int`.
- `config/memory.yaml` current contents: `namespaces` = `{episodic: {ttl_days: 90, cap: 50},
  member: {ttl_days: 365, cap: 100}, provider: {ttl_days: 365, cap: 100},
  policy_notes: {ttl_days: null, cap: 200}}`; `importance_weights` = `{routine: 1.0,
  notable: 1.5, critical: 3.0}`; `recency_half_life_days: 30`.
- `pa_copilot.rag.embedder.BgeEmbedder(model_name: str, query_prefix: str = "")` — lazy CPU
  `sentence-transformers`; `.embed_documents(list[str]) -> list[list[float]]`,
  `.embed_query(str) -> list[float]`. `EMBED_DIM_BGE_SMALL = 384`.
- `tests/_fakes.py::FakeEmbedder` — `DIM = 64`; `.embed_documents`, `.embed_query`
  (crc32-bucketed, normalized, per-process stable). Not a LangChain `Embeddings`, not
  callable — reached only through `memory.embeddings.as_embeddings(...)`.
- `pa_copilot.state.PACaseState` — `TypedDict(total=False)`; `working_memory: dict[str, Any]`
  (last-write-wins, seeded `{}` by `new_case_state`), `messages` (`add_messages`).
- `tests/conftest.py` fixtures: `_clear_settings_cache` (autouse), `tmp_trace_dir`,
  `frozen_now` (`"2026-09-09T12:00:00+00:00"`), `sample_request`.

## Library facts verified for this plan (2026-09-10, installed versions)

- `SqliteStore.from_conn_string(conn_string, *, index=None, ttl=None)` is a
  `@contextmanager` that **closes the connection on exit** — unusable for a long-lived
  store. Construct directly:
  `SqliteStore(sqlite3.connect(path, check_same_thread=False, isolation_level=None),
  index=<SqliteIndexConfig|None>, ttl=<TTLConfig|None>)`, then call `store.setup()` once.
- `store.setup()` runs `conn.enable_load_extension(True); sqlite_vec.load(conn)` **only when
  `index` is provided**. That call is the sqlite-vec failure point to guard.
- `SqliteStore` does **not** override `put` / `aput` — `BaseStore.put(namespace, key, value,
  index=None, *, ttl=NOT_PROVIDED)` builds a `PutOp` and calls `self.batch([op])`.
  Overriding `put` / `aput` on the subclass therefore intercepts LangMem tool writes (they
  call `store.put` / `store.aput`). Direct `store.batch([...])` calls are NOT intercepted —
  first-party code must use `put`/`aput`; document this.
- TTL sentinel import: `from langgraph.store.base import NOT_PROVIDED`.
  `_ensure_ttl(cfg, ttl)`: an explicit `ttl=None` always means "no expiry for this item"
  regardless of the store's `TTLConfig`; `ttl=<minutes>` sets a per-item expiry.
- `TTLConfig` is a `TypedDict`: `refresh_on_read: bool`, `default_ttl: float | None`
  (minutes, single global — cannot express per-namespace), `sweep_interval_minutes: int | None`.
- `store.search(namespace_prefix, /, *, query=None, filter=None, limit=10, offset=0,
  refresh_ttl=None) -> list[SearchItem]`. `SearchItem` attrs: `.namespace`, `.key`,
  `.value` (dict), `.created_at` (`datetime`), `.updated_at` (`datetime`), `.score`
  (`float | None` — similarity, higher is better; `None` when `query` is `None`).
- `store.sweep_ttl() -> int` deletes rows where `expires_at IS NOT NULL AND expires_at <
  CURRENT_TIMESTAMP`; returns the delete count. Column is `store.expires_at`.
- `store.list_namespaces(*, prefix=None, suffix=None, max_depth=None, limit=100, offset=0)
  -> list[tuple[str, ...]]`.
- `langmem.create_manage_memory_tool(namespace, *, store=None, actions_permitted=('create',
  'update','delete'), name='manage_memory', ...)` and
  `create_search_memory_tool(namespace, *, store=None, name='search_memory', ...)` — both
  accept an explicit `store=`, so they work standalone without a compiled graph.
- `langchain_core.embeddings.Embeddings` — ABC with abstract `embed_documents` /
  `embed_query`; subclass directly.

## File Structure

| File | Responsibility |
|---|---|
| `src/pa_copilot/memory/__init__.py` | package marker; re-export `open_memory_store`, `build_memory_tools`, `PolicyStore` |
| `src/pa_copilot/memory/embeddings.py` | `LocalEmbeddings(Embeddings)` wrapping `BgeEmbedder`; `as_embeddings(obj)` duck-adapter; module-cached default |
| `src/pa_copilot/memory/policy.py` | pure policy math: `namespace_policy`, `ttl_minutes_for`, `importance_of`, `importance_weight`, `recency_decay`, `rank_key`, `search_ranked`, `enforce_cap`, `sweep_expired` |
| `src/pa_copilot/memory/store.py` | `PolicyStore(SqliteStore)` (put/aput override + cap); `open_memory_store(...)` factory with sqlite-vec fallback; `memory_store(...)` test contextmanager |
| `src/pa_copilot/memory/tools.py` | `build_memory_tools(store, namespace) -> (manage, search)` over LangMem |
| `src/pa_copilot/memory/working.py` | Tier-1 helpers: `remember`, `recall`, `search_working` over `state["working_memory"]` |
| `src/pa_copilot/config.py` | `MemoryConfig` gains `default_importance`, `semantic_fields`; `_build_memory_config` reads them |
| `config/memory.yaml` | add `default_importance: routine`, `semantic_fields: [content, text]` |
| `tests/conftest.py` | add autouse `_env_snapshot`; `fake_embedder`, `memory_store` fixtures |
| `scripts/memory_demo.py` | in-repo producer of `traces/tiered_memory_recall.json` (deterministic, fake embedder) |
| `scripts/run_persistence_test.py` | AC-07 two-process driver → `traces/memory_persistence.log` |
| `docs/memory-policy.md` | the policy write-up (AC-08 evidence) |
| `tests/test_memory_embeddings.py` | `as_embeddings` adapter + dims |
| `tests/test_memory_policy.py` | policy scalar math |
| `tests/test_memory_store.py` | `PolicyStore` put injection, cap enforcement, sqlite-vec fallback |
| `tests/test_memory_tools.py` | LangMem manage/search smoke over `PolicyStore` |
| `tests/test_memory_working.py` | Tier-1 helpers |
| `tests/test_env_isolation.py` | `_env_snapshot` reverts a raw `os.environ` write |
| `tests/test_ac06_tiered_memory.py` | AC-06: working-memory turn recall + long-term semantic recall + trace guard |
| `tests/test_ac08_eviction.py` | AC-08: overflow evicts stale low-importance, keeps `critical`, sweep clears expired |
| `tests/test_memory_persistence.py` | AC-07: in-process teardown+rebuild + subprocess script + log guard |
| `Makefile` | retarget `persistence-test` to `python scripts/run_persistence_test.py` |
| `docs/design.md` | §6.3 note that per-namespace TTL is injected at `put` time by `PolicyStore`; §1 row mentions `PolicyStore` |
| `specs/acceptance-criteria.md` | AC-06/07/08 rows → `done` |
| `docs/rubric-coverage.md` | Memory Systems row → PR4 / done |

---

### Task 1: Test-environment isolation fixture

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_config.py` (one comment only — see Step 5)
- Test: `tests/test_env_isolation.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: an autouse fixture `_env_snapshot` in `tests/conftest.py` that snapshots
  `os.environ` before each test and restores it exactly afterward (removes keys added during
  the test, restores changed/removed keys). No public Python symbol.

**Why:** PR2 parked this (Ruling R19). `config.load_settings` does a raw
`os.environ["GOOGLE_API_KEY"] = ...` mirror write that `monkeypatch` cannot revert, so
`test_config.py::test_gemini_api_key_mirrors_to_google_api_key` currently leaks
`GOOGLE_API_KEY` into the rest of the pytest session. PR4's store/config tests set several
`PA_*` env vars — they need a guaranteed-clean environ per test.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env_isolation.py
"""The autouse _env_snapshot fixture (tests/conftest.py) must revert any raw
os.environ mutation a test makes — monkeypatch alone cannot, and config.load_settings
does a raw os.environ write. Two tests in definition order: the first leaks, the
second proves the leak was reverted."""

import os


def test_aaa_writes_a_raw_environ_key():
    os.environ["PA_TEST_LEAK_CHECK"] = "leaked"
    assert os.environ["PA_TEST_LEAK_CHECK"] == "leaked"


def test_bbb_raw_environ_key_was_reverted():
    assert "PA_TEST_LEAK_CHECK" not in os.environ
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_env_isolation.py -v`
Expected: `test_bbb_raw_environ_key_was_reverted` FAILS (`PA_TEST_LEAK_CHECK` still set).

- [ ] **Step 3: Add the fixture**

```python
# tests/conftest.py — add near the top, after the imports
import os


@pytest.fixture(autouse=True)
def _env_snapshot():
    """Snapshot os.environ and restore it exactly after each test. monkeypatch
    reverts its own setenv/delenv, but not a raw `os.environ[...] = ...` write
    (config.load_settings mirrors GEMINI_API_KEY -> GOOGLE_API_KEY that way)."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        for key in [k for k in os.environ if k not in saved]:
            del os.environ[key]
        for key, value in saved.items():
            if os.environ.get(key) != value:
                os.environ[key] = value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_env_isolation.py -v`
Expected: both PASS.

- [ ] **Step 5: Drop the now-redundant workaround note and re-run the config suite**

In `tests/test_config.py::test_gemini_api_key_mirrors_to_google_api_key`, the
`monkeypatch.delenv("GOOGLE_API_KEY", raising=False)` line stays (it sets up the
precondition), but add a one-line comment above it:
`# _env_snapshot (conftest) reverts the raw mirror write after this test`.

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS, pristine.

- [ ] **Step 6: Full fast suite stays green**

Run: `python -m pytest -q -m "not slow"`
Expected: all PASS (90 prior + 2 new), pristine output.

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py tests/test_env_isolation.py tests/test_config.py
git commit -m "test: autouse os.environ snapshot/restore fixture (PR2 park)"
```

---

### Task 2: Config extension + policy scalar math

**Files:**
- Modify: `src/pa_copilot/config.py` (`MemoryConfig`, `_build_memory_config`)
- Modify: `config/memory.yaml`
- Create: `src/pa_copilot/memory/__init__.py`
- Create: `src/pa_copilot/memory/policy.py`
- Test: `tests/test_memory_policy.py` (create); `tests/test_config.py` (add cases)

**Interfaces:**
- Consumes: `pa_copilot.config.MemoryConfig`, `NamespacePolicy`, `get_settings`.
- Produces (all in `pa_copilot.memory.policy`, `mem` is a `MemoryConfig`):
  - `namespace_policy(namespace: tuple[str, ...], mem: MemoryConfig) -> NamespacePolicy | None`
    — keys off `namespace[1]` (`("pa","member","M1")` and `("pa","episodic")` both resolve);
    `None` when the namespace is not governed.
  - `ttl_minutes_for(namespace: tuple[str, ...], mem: MemoryConfig) -> float | None` —
    `pol.ttl_days * 24 * 60`, or `None` (no expiry) when the policy or its `ttl_days` is `None`.
  - `importance_of(value: dict, mem: MemoryConfig) -> str` — `value.get("importance")` if a
    known weight key, else `mem.default_importance`.
  - `importance_weight(value: dict, mem: MemoryConfig) -> float` —
    `mem.importance_weights.get(importance_of(value, mem), 1.0)`.
  - `recency_decay(ts, *, now: datetime, half_life_days: float) -> float` — `ts` is a
    `datetime` or ISO string; `0.5 ** (age_days / half_life_days)`, clamped to `(0, 1]`,
    `1.0` when `ts` is in the future or unparseable.
  - `rank_key(item, mem: MemoryConfig, *, now: datetime) -> float` — `item` is anything with
    `.value`, `.updated_at` (fallback `.created_at`), optional `.score`;
    `(item.score or 1.0) * importance_weight(item.value, mem) * recency_decay(...)`.
  - `MemoryConfig.default_importance: str` (`"routine"`), `MemoryConfig.semantic_fields:
    tuple[str, ...]` (`("content", "text")`).
- `search_ranked` / `enforce_cap` / `sweep_expired` are declared here but IMPLEMENTED AND
  TESTED in Task 4 (they need a live store) — add them as `def ...: raise NotImplementedError`
  stubs now with the signatures in Task 4's Interfaces block, or omit and let Task 4 add
  them. Prefer: omit now, Task 4 adds them.

- [ ] **Step 1: Write the failing config test**

```python
# tests/test_config.py — add
def test_memory_config_has_policy_defaults(config_dir):
    from pa_copilot.config import load_settings
    m = load_settings(config_dir=config_dir, env_file=None).memory
    assert m.default_importance == "routine"
    assert "content" in m.semantic_fields and "text" in m.semantic_fields
    assert m.namespaces["episodic"].ttl_days == 90
    assert m.namespaces["policy_notes"].ttl_days is None
```

- [ ] **Step 2: Run it red**

Run: `python -m pytest tests/test_config.py::test_memory_config_has_policy_defaults -v`
Expected: FAIL (`AttributeError: ... 'default_importance'`).

- [ ] **Step 3: Extend `MemoryConfig` + yaml**

`config/memory.yaml` — append:

```yaml
default_importance: routine
semantic_fields: [content, text]
```

`src/pa_copilot/config.py` — `MemoryConfig` (insert the two fields **before** `raw` so the
defaulted `raw` stays last):

```python
@dataclass(frozen=True)
class MemoryConfig:
    namespaces: dict[str, NamespacePolicy]
    importance_weights: dict[str, float]
    recency_half_life_days: int
    default_importance: str = "routine"
    semantic_fields: tuple[str, ...] = ("content", "text")
    raw: dict = field(default_factory=dict)
```

`_build_memory_config`:

```python
    return MemoryConfig(
        namespaces=namespaces,
        importance_weights=weights,
        recency_half_life_days=int(raw.get("recency_half_life_days", 30)),
        default_importance=str(raw.get("default_importance", "routine")),
        semantic_fields=tuple(raw.get("semantic_fields") or ("content", "text")),
        raw=raw,
    )
```

- [ ] **Step 4: Run it green**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Write the failing policy test**

```python
# tests/test_memory_policy.py
"""AC-08 policy math — pure functions, no store, no model."""

from datetime import datetime, timedelta, timezone

import pytest

from pa_copilot.config import get_settings
from pa_copilot.memory import policy

NOW = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mem():
    return get_settings().memory


def test_namespace_policy_resolves_both_shapes(mem):
    assert policy.namespace_policy(("pa", "member", "M1"), mem).cap == 100
    assert policy.namespace_policy(("pa", "episodic"), mem).ttl_days == 90
    assert policy.namespace_policy(("pa", "unknown"), mem) is None


def test_ttl_minutes_for(mem):
    assert policy.ttl_minutes_for(("pa", "episodic"), mem) == 90 * 24 * 60
    assert policy.ttl_minutes_for(("pa", "policy_notes"), mem) is None
    assert policy.ttl_minutes_for(("pa", "unknown"), mem) is None


def test_importance(mem):
    assert policy.importance_of({}, mem) == "routine"
    assert policy.importance_of({"importance": "critical"}, mem) == "critical"
    assert policy.importance_weight({"importance": "critical"}, mem) == 3.0
    assert policy.importance_weight({}, mem) == 1.0


def test_recency_decay_halves_at_half_life(mem):
    older = NOW - timedelta(days=mem.recency_half_life_days)
    assert policy.recency_decay(older, now=NOW, half_life_days=mem.recency_half_life_days) == pytest.approx(0.5, rel=1e-6)
    assert policy.recency_decay(NOW, now=NOW, half_life_days=30) == pytest.approx(1.0)
    assert policy.recency_decay("not-a-date", now=NOW, half_life_days=30) == 1.0


def test_rank_key_combines_factors(mem):
    class Item:
        value = {"importance": "critical"}
        updated_at = NOW
        created_at = NOW
        score = 0.5

    assert policy.rank_key(Item(), mem, now=NOW) == pytest.approx(0.5 * 3.0 * 1.0)
```

- [ ] **Step 6: Run it red**

Run: `python -m pytest tests/test_memory_policy.py -v`
Expected: FAIL (`ModuleNotFoundError: pa_copilot.memory`).

- [ ] **Step 7: Implement `memory/__init__.py` + `memory/policy.py`**

`src/pa_copilot/memory/__init__.py`:

```python
"""Tiered memory: Tier-1 working scratch (working.py) + Tier-2 long-term
semantic store (store.py) governed by policy.py, with LangMem tools (tools.py)."""

from pa_copilot.memory.store import PolicyStore, open_memory_store
from pa_copilot.memory.tools import build_memory_tools

__all__ = ["PolicyStore", "open_memory_store", "build_memory_tools"]
```

> NOTE: the `__init__` re-exports reference Task 3/4 modules. Create `__init__.py` with the
> docstring only in this task; add the imports in Task 4's final step. (Keeps Task 2's
> import graph clean.)

`src/pa_copilot/memory/policy.py`:

```python
"""Memory policy: per-namespace TTL, importance weighting, recency decay, and the
combined rank key used for LRU-cap eviction and ranked search (design.md §6.3).

Pure functions — a store handle is passed in where one is needed (Task 4). The
`PolicyStore` (store.py) is what actually calls `ttl_minutes_for` on every write."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Protocol

from pa_copilot.config import MemoryConfig, NamespacePolicy


class _Ranked(Protocol):
    value: dict[str, Any]
    updated_at: Any
    created_at: Any


def namespace_policy(namespace: tuple[str, ...], mem: MemoryConfig) -> NamespacePolicy | None:
    key = namespace[1] if len(namespace) > 1 else (namespace[0] if namespace else None)
    return mem.namespaces.get(key) if key else None


def ttl_minutes_for(namespace: tuple[str, ...], mem: MemoryConfig) -> float | None:
    pol = namespace_policy(namespace, mem)
    if pol is None or pol.ttl_days is None:
        return None
    return float(pol.ttl_days) * 24.0 * 60.0


def importance_of(value: dict[str, Any], mem: MemoryConfig) -> str:
    label = (value or {}).get("importance")
    return label if label in mem.importance_weights else mem.default_importance


def importance_weight(value: dict[str, Any], mem: MemoryConfig) -> float:
    return float(mem.importance_weights.get(importance_of(value, mem), 1.0))


def _as_datetime(ts: Any) -> datetime | None:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, str):
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def recency_decay(ts: Any, *, now: datetime, half_life_days: float) -> float:
    when = _as_datetime(ts)
    if when is None:
        return 1.0
    age_days = (now - when).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    return float(0.5 ** (age_days / max(half_life_days, 1e-9)))


def rank_key(item: _Ranked, mem: MemoryConfig, *, now: datetime) -> float:
    score = getattr(item, "score", None)
    sem = float(score) if score is not None else 1.0
    ts = getattr(item, "updated_at", None) or getattr(item, "created_at", None)
    return (
        sem
        * importance_weight(getattr(item, "value", {}) or {}, mem)
        * recency_decay(ts, now=now, half_life_days=mem.recency_half_life_days)
    )
```

(`math` import is used by nothing yet — drop it if `ruff` complains; keep the module clean.)

- [ ] **Step 8: Run it green**

Run: `python -m pytest tests/test_memory_policy.py tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 9: Lint + full fast suite**

Run: `ruff check src tests && python -m pytest -q -m "not slow"`
Expected: clean; all PASS pristine.

- [ ] **Step 10: Commit**

```bash
git add src/pa_copilot/config.py config/memory.yaml src/pa_copilot/memory/__init__.py \
        src/pa_copilot/memory/policy.py tests/test_memory_policy.py tests/test_config.py
git commit -m "feat(memory): config policy fields + policy scalar math (AC-08)"
```

---

### Task 3: Embeddings adapter

**Files:**
- Create: `src/pa_copilot/memory/embeddings.py`
- Test: `tests/test_memory_embeddings.py` (create)

**Interfaces:**
- Consumes: `pa_copilot.rag.embedder.BgeEmbedder`, `EMBED_DIM_BGE_SMALL`;
  `pa_copilot.config.get_settings`; `langchain_core.embeddings.Embeddings`.
- Produces (in `pa_copilot.memory.embeddings`):
  - `class LocalEmbeddings(Embeddings)` — `__init__(self, model_name: str | None = None,
    query_prefix: str | None = None)`; defaults pulled from `get_settings()`. Wraps a
    module-cached `BgeEmbedder` (build the model once — see `rag/tool.py::_default_embedder`
    for the pattern). `.embed_documents`, `.embed_query`.
  - `as_embeddings(obj) -> Embeddings` — passthrough if already an `Embeddings`; else if
    `obj` has `.embed_documents` and `.embed_query`, wrap in a private `_DuckEmbeddings`;
    else `TypeError`.
  - `embedding_dims(obj) -> int` — `len(as_embeddings(obj).embed_query("dimension probe"))`.
    (No magic constant — works for `FakeEmbedder` (64) and bge-small (384) alike. One extra
    embed call at store-open; acceptable.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_embeddings.py
import pytest

from pa_copilot.memory import embeddings as E

pytestmark = pytest.mark.filterwarnings("ignore")  # remove if not needed


def test_as_embeddings_wraps_a_duck(fake_embedder):
    emb = E.as_embeddings(fake_embedder)
    assert hasattr(emb, "embed_documents") and hasattr(emb, "embed_query")
    v = emb.embed_query("hello world")
    assert isinstance(v, list) and len(v) == fake_embedder.DIM


def test_as_embeddings_passthrough(fake_embedder):
    wrapped = E.as_embeddings(fake_embedder)
    assert E.as_embeddings(wrapped) is wrapped


def test_as_embeddings_rejects_junk():
    with pytest.raises(TypeError):
        E.as_embeddings(object())


def test_embedding_dims(fake_embedder):
    assert E.embedding_dims(fake_embedder) == 64


@pytest.mark.slow
def test_local_embeddings_real_bge_dims():
    assert E.embedding_dims(E.LocalEmbeddings()) == 384
```

Add the `fake_embedder` fixture to `tests/conftest.py` in this task:

```python
# tests/conftest.py
@pytest.fixture
def fake_embedder():
    from tests._fakes import FakeEmbedder  # noqa: PLC0415
    return FakeEmbedder()
```

> `tests/` is not a package. `from tests._fakes import FakeEmbedder` works because
> `pyproject` sets `pythonpath = ["."]` and PR3 already imports `_fakes` this way
> (`tests/test_ac11_agentic_rag.py`). If the bare import fails in the fixture, use
> `import _fakes; return _fakes.FakeEmbedder()`.

- [ ] **Step 2: Run it red**

Run: `python -m pytest tests/test_memory_embeddings.py -v -m "not slow"`
Expected: FAIL (`ModuleNotFoundError` / `AttributeError`).

- [ ] **Step 3: Implement `memory/embeddings.py`**

```python
"""Adapt any duck-typed embedder to the LangChain `Embeddings` interface that
`SqliteStore`'s index config wants. The real model is bge-small (reused from
`rag/`); tests pass `tests/_fakes.py::FakeEmbedder` through `as_embeddings`."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from pa_copilot.config import get_settings
from pa_copilot.rag.embedder import BgeEmbedder

_default_bge: BgeEmbedder | None = None


def _shared_bge(model_name: str, query_prefix: str) -> BgeEmbedder:
    global _default_bge
    if _default_bge is None or _default_bge.model_name != model_name:
        _default_bge = BgeEmbedder(model_name, query_prefix=query_prefix)
    return _default_bge


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str | None = None, query_prefix: str | None = None) -> None:
        s = get_settings()
        self._bge = _shared_bge(
            model_name or s.embedding_model,
            query_prefix if query_prefix is not None else s.rag_query_prefix,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._bge.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._bge.embed_query(text)


class _DuckEmbeddings(Embeddings):
    def __init__(self, inner: object) -> None:
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


def as_embeddings(obj: object) -> Embeddings:
    if isinstance(obj, Embeddings):
        return obj
    if hasattr(obj, "embed_documents") and hasattr(obj, "embed_query"):
        return _DuckEmbeddings(obj)
    raise TypeError(f"{obj!r} is not an Embeddings and has no embed_documents/embed_query")


def embedding_dims(obj: object) -> int:
    return len(as_embeddings(obj).embed_query("dimension probe"))
```

- [ ] **Step 4: Run it green**

Run: `python -m pytest tests/test_memory_embeddings.py -q -m "not slow"`
Expected: PASS.

- [ ] **Step 5: Lint + fast suite + commit**

Run: `ruff check src tests && python -m pytest -q -m "not slow"`

```bash
git add src/pa_copilot/memory/embeddings.py tests/test_memory_embeddings.py tests/conftest.py
git commit -m "feat(memory): LangChain Embeddings adapter over BgeEmbedder / fakes"
```

---

### Task 4: `PolicyStore` + `open_memory_store` + sqlite-vec fallback + store-coupled policy ops

**Files:**
- Create: `src/pa_copilot/memory/store.py`
- Modify: `src/pa_copilot/memory/policy.py` (add `search_ranked`, `enforce_cap`, `sweep_expired`)
- Modify: `tests/conftest.py` (add `memory_store` fixture)
- Test: `tests/test_memory_store.py` (create)

**Interfaces:**
- Consumes: `SqliteStore` (`langgraph.store.sqlite`), `TTLConfig`, `NOT_PROVIDED`,
  `SearchItem` (`langgraph.store.base`); `pa_copilot.config.get_settings`, `Settings`;
  `pa_copilot.memory.policy` (`ttl_minutes_for`, `importance_of`, `rank_key`,
  `namespace_policy`); `pa_copilot.memory.embeddings` (`as_embeddings`, `embedding_dims`).
- Produces:
  - `pa_copilot.memory.store.PolicyStore(SqliteStore)` —
    `__init__(self, conn, *, settings: Settings | None = None, index=None, ttl=None)`;
    overrides `put(namespace, key, value, index=None, *, ttl=NOT_PROVIDED) -> None` and
    `async aput(...) -> None` to: (1) `value = {**value, "importance": policy.importance_of(
    value, mem)}` when `"importance"` absent, (2) if `ttl is NOT_PROVIDED`, set
    `ttl = policy.ttl_minutes_for(namespace, mem)`, (3) call `super().put/aput(...)`,
    (4) `policy.enforce_cap(self, namespace, mem=mem)`. Property
    `semantic_index_available -> bool` (`self.index_config is not None`). Attr
    `semantic_error: str | None` (set by the factory on fallback).
  - `pa_copilot.memory.store.open_memory_store(path: str | None = None, *,
    embedder: object | None = None, settings: Settings | None = None,
    semantic: bool = True) -> PolicyStore` — builds `sqlite3.connect(path,
    check_same_thread=False, isolation_level=None)`; when `semantic and embedder is not
    None`, an index config `{"dims": embedding_dims(embedder), "embed":
    as_embeddings(embedder), "fields": list(settings.memory.semantic_fields)}`; constructs
    `PolicyStore`, calls `.setup()`. If `.setup()` raises with an index configured
    (sqlite-vec load failure), closes the conn, rebuilds with `index=None`, records
    `store.semantic_error = repr(exc)`, retries `.setup()`. Returns the store (caller owns
    `.conn.close()`).
  - `pa_copilot.memory.store.memory_store(...)` — a `@contextlib.contextmanager` with the
    same signature that yields the store and closes `store.conn` on exit (test/CLI use).
  - Added to `pa_copilot.memory.policy`:
    - `search_ranked(store, namespace: tuple[str, ...], query: str | None, *,
      limit: int | None = None, mem: MemoryConfig | None = None,
      now: datetime | None = None, pool: int = 50) -> list[SearchItem]` — calls
      `store.search(namespace, query=query if store.semantic_index_available else None,
      limit=max(pool, limit or 0))`, re-sorts by `rank_key` desc, returns `[:limit]`.
    - `enforce_cap(store, namespace: tuple[str, ...], *, mem: MemoryConfig | None = None,
      now: datetime | None = None) -> list[str]` — no-op when the namespace has no cap or is
      under it; else deletes the lowest-`rank_key` non-`critical` items until at cap (or
      only `critical` remain); returns evicted keys.
    - `sweep_expired(store) -> int` — `return store.sweep_ttl()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memory_store.py
"""PolicyStore write interception (AC-08), LRU cap, and the sqlite-vec fallback."""

from datetime import datetime, timezone

import pytest

from pa_copilot.config import get_settings
from pa_copilot.memory import policy
from pa_copilot.memory.store import PolicyStore, memory_store, open_memory_store

NS_MEMBER = ("pa", "member", "M100001")
NS_NOTES = ("pa", "policy_notes")


def test_put_injects_namespace_ttl_and_default_importance(memory_store):
    memory_store.put(NS_MEMBER, "rec1", {"content": "8 weeks PT completed"})
    item = memory_store.get(NS_MEMBER, "rec1")
    assert item.value["importance"] == "routine"
    # member namespace = 365d TTL -> expires_at is set on the row
    row = memory_store.conn.execute(
        "SELECT expires_at FROM store WHERE key = ?", ("rec1",)
    ).fetchone()
    assert row[0] is not None


def test_put_respects_no_ttl_namespace(memory_store):
    memory_store.put(NS_NOTES, "n1", {"content": "policy PA-PSG requires in-lab first"})
    row = memory_store.conn.execute(
        "SELECT expires_at FROM store WHERE key = ?", ("n1",)
    ).fetchone()
    assert row[0] is None


def test_explicit_importance_is_kept(memory_store):
    memory_store.put(NS_MEMBER, "denial1", {"content": "denied", "importance": "critical"})
    assert memory_store.get(NS_MEMBER, "denial1").value["importance"] == "critical"


def test_lru_cap_evicts_lowest_rank_keeps_critical(tmp_path, fake_embedder, monkeypatch):
    # shrink the member cap to 3 for the test
    s = get_settings()
    monkeypatch.setitem(s.memory.namespaces, "member",
                        type(s.memory.namespaces["member"])(ttl_days=365, cap=3))
    with memory_store(tmp_path / "m.db", embedder=fake_embedder) as store:
        store.put(NS_MEMBER, "old_routine", {"content": "routine note one"})
        store.put(NS_MEMBER, "crit", {"content": "prior denial", "importance": "critical"})
        store.put(NS_MEMBER, "b", {"content": "note b"})
        store.put(NS_MEMBER, "c", {"content": "note c"})   # 4 > cap 3 -> evict one
        keys = {i.key for i in store.search(NS_MEMBER, limit=50)}
        assert "crit" in keys
        assert len(keys) == 3


def test_sqlite_vec_fallback(tmp_path, fake_embedder, monkeypatch):
    import sqlite_vec

    def boom(conn):
        raise RuntimeError("sqlite-vec unavailable")

    monkeypatch.setattr(sqlite_vec, "load", boom)
    with memory_store(tmp_path / "fb.db", embedder=fake_embedder) as store:
        assert store.semantic_index_available is False
        assert "sqlite-vec unavailable" in store.semantic_error
        store.put(NS_MEMBER, "x", {"content": "still works without a vector index"})
        got = policy.search_ranked(store, NS_MEMBER, "anything", limit=5)
        assert [i.key for i in got] == ["x"]


def test_search_ranked_orders_by_importance_then_recency(memory_store):
    memory_store.put(NS_MEMBER, "routine1", {"content": "routine lumbar note"})
    memory_store.put(NS_MEMBER, "crit1", {"content": "routine lumbar note", "importance": "critical"})
    ranked = policy.search_ranked(memory_store, NS_MEMBER, "lumbar note", limit=2)
    assert ranked[0].key == "crit1"
```

- [ ] **Step 2: Run red**

Run: `python -m pytest tests/test_memory_store.py -v -m "not slow"`
Expected: FAIL (`ImportError: cannot import name 'PolicyStore'`).

- [ ] **Step 3: Implement `memory/store.py`**

```python
"""Long-term semantic memory: a SqliteStore subclass that enforces the design.md
§6.3 policy on every write, plus a factory that degrades to a non-semantic store
when sqlite-vec will not load."""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator

from langgraph.store.base import NOT_PROVIDED, TTLConfig
from langgraph.store.sqlite import SqliteStore

from pa_copilot.config import Settings, get_settings
from pa_copilot.memory import policy
from pa_copilot.memory.embeddings import as_embeddings, embedding_dims


class PolicyStore(SqliteStore):
    """Every put/aput gets the namespace's TTL and a default importance stamped
    on, then the namespace's LRU cap enforced. LangMem's manage_memory tool calls
    `store.put`, so its writes are governed too. NOTE: a raw `store.batch([...])`
    bypasses this — first-party code uses put/aput."""

    def __init__(self, conn, *, settings: Settings | None = None, index=None, ttl=None):
        super().__init__(conn, index=index, ttl=ttl)
        self._settings = settings or get_settings()
        self.semantic_error: str | None = None

    @property
    def _mem(self):
        return self._settings.memory

    @property
    def semantic_index_available(self) -> bool:
        return self.index_config is not None

    def _prep(self, namespace, value, ttl):
        if "importance" not in (value or {}):
            value = {**(value or {}), "importance": policy.importance_of(value or {}, self._mem)}
        if ttl is NOT_PROVIDED:
            ttl = policy.ttl_minutes_for(tuple(namespace), self._mem)
        return value, ttl

    def put(self, namespace, key, value, index=None, *, ttl=NOT_PROVIDED) -> None:
        value, ttl = self._prep(namespace, value, ttl)
        super().put(namespace, key, value, index=index, ttl=ttl)
        policy.enforce_cap(self, tuple(namespace), mem=self._mem)

    async def aput(self, namespace, key, value, index=None, *, ttl=NOT_PROVIDED) -> None:
        value, ttl = self._prep(namespace, value, ttl)
        await super().aput(namespace, key, value, index=index, ttl=ttl)
        policy.enforce_cap(self, tuple(namespace), mem=self._mem)


def _build(path: str, index, settings: Settings) -> PolicyStore:
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    ttl = TTLConfig(refresh_on_read=True)
    return PolicyStore(conn, settings=settings, index=index, ttl=ttl)


def open_memory_store(
    path: str | None = None,
    *,
    embedder: object | None = None,
    settings: Settings | None = None,
    semantic: bool = True,
) -> PolicyStore:
    s = settings or get_settings()
    path = path or s.memory_db
    index = None
    if semantic and embedder is not None:
        index = {
            "dims": embedding_dims(embedder),
            "embed": as_embeddings(embedder),
            "fields": list(s.memory.semantic_fields),
        }
    store = _build(str(path), index, s)
    try:
        store.setup()
    except Exception as exc:  # noqa: BLE001 -- sqlite-vec extension load failure
        if index is None:
            raise
        store.conn.close()
        store = _build(str(path), None, s)
        store.semantic_error = repr(exc)
        store.setup()
    return store


@contextlib.contextmanager
def memory_store(
    path: str | None = None,
    *,
    embedder: object | None = None,
    settings: Settings | None = None,
    semantic: bool = True,
) -> Iterator[PolicyStore]:
    store = open_memory_store(path, embedder=embedder, settings=settings, semantic=semantic)
    try:
        yield store
    finally:
        store.conn.close()
```

- [ ] **Step 4: Add `search_ranked` / `enforce_cap` / `sweep_expired` to `memory/policy.py`**

```python
# append to src/pa_copilot/memory/policy.py

def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def search_ranked(store, namespace, query, *, limit=None, mem=None, now=None, pool=50):
    mem = mem or _settings_mem()
    when = _now(now)
    use_query = query if getattr(store, "semantic_index_available", True) else None
    raw = store.search(namespace, query=use_query, limit=max(pool, limit or 0))
    raw.sort(key=lambda it: rank_key(it, mem, now=when), reverse=True)
    return raw[:limit] if limit else raw


def enforce_cap(store, namespace, *, mem=None, now=None):
    mem = mem or _settings_mem()
    pol = namespace_policy(namespace, mem)
    if not pol or not pol.cap:
        return []
    items = store.search(namespace, limit=10_000)
    overflow = len(items) - pol.cap
    if overflow <= 0:
        return []
    when = _now(now)
    items.sort(key=lambda it: rank_key(it, mem, now=when))  # worst first
    evicted: list[str] = []
    for it in items:
        if overflow <= 0:
            break
        if importance_of(it.value or {}, mem) == "critical":
            continue
        store.delete(namespace, it.key)
        evicted.append(it.key)
        overflow -= 1
    return evicted


def sweep_expired(store) -> int:
    return store.sweep_ttl()


def _settings_mem():
    from pa_copilot.config import get_settings
    return get_settings().memory
```

- [ ] **Step 5: Add the `memory_store` fixture to `tests/conftest.py`**

```python
@pytest.fixture
def memory_store(tmp_path, fake_embedder):
    from pa_copilot.memory.store import memory_store as _open
    with _open(tmp_path / "memory.db", embedder=fake_embedder) as store:
        yield store
```

> The fixture name shadows the `memory_store` contextmanager inside test modules that
> import both — tests either use the fixture OR import the CM under an alias. Test files in
> this plan that need the CM (`test_memory_store.py`) import it and also use the fixture;
> that is fine because the fixture is injected by name and the import is only referenced
> explicitly. If `ruff`/pytest flags a collision, rename the fixture to `mem_store` and
> update references.

- [ ] **Step 6: Run green**

Run: `python -m pytest tests/test_memory_store.py tests/test_memory_policy.py -q -m "not slow"`
Expected: PASS.

- [ ] **Step 7: Lint + full fast suite (pristine)**

Run: `ruff check src tests && python -m pytest -q -m "not slow"`
Expected: clean; all PASS; **no new warnings**. If `langgraph.store.sqlite` import emits a
warning caught by `filterwarnings=["error"]`, add a **narrow** `category + module`-scoped
ignore to `pyproject.toml` `filterwarnings` (mirror the existing chromadb entries) and note
it in the commit body — do not broaden to a bare `ignore`.

- [ ] **Step 8: Commit**

```bash
git add src/pa_copilot/memory/store.py src/pa_copilot/memory/policy.py \
        tests/test_memory_store.py tests/conftest.py pyproject.toml
git commit -m "feat(memory): PolicyStore write-policy + sqlite-vec fallback (AC-08)"
```

---

### Task 5: LangMem tools + Tier-1 working memory

**Files:**
- Create: `src/pa_copilot/memory/tools.py`
- Create: `src/pa_copilot/memory/working.py`
- Modify: `src/pa_copilot/memory/__init__.py` (add the re-export imports deferred in Task 2)
- Test: `tests/test_memory_tools.py`, `tests/test_memory_working.py` (create)

**Interfaces:**
- Consumes: `langmem.create_manage_memory_tool`, `create_search_memory_tool`;
  `pa_copilot.memory.store.PolicyStore`.
- Produces:
  - `pa_copilot.memory.tools.build_memory_tools(store, namespace: tuple[str, ...] | str, *,
    manage_instructions: str | None = None) -> tuple[BaseTool, BaseTool]` — returns
    `(manage_memory, search_memory)` bound to `store` and `namespace`. `namespace` may
    contain a `"{member_id}"` placeholder (LangMem substitutes from the runtime config in
    PR5); PR4 passes concrete tuples.
  - `pa_copilot.memory.working.remember(working_memory: dict, key: str, value) -> dict` —
    returns a new dict with `value` recorded under `working_memory["facts"][key]` (does not
    mutate the input).
  - `pa_copilot.memory.working.recall(working_memory: dict, key: str, default=None)` —
    `working_memory["facts"].get(key, default)`.
  - `pa_copilot.memory.working.search_working(working_memory: dict, substring: str) ->
    list[tuple[str, object]]` — `(key, value)` pairs whose key or `str(value)` contains
    `substring` (case-insensitive).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memory_working.py
from pa_copilot.memory import working


def test_remember_is_immutable_and_recallable():
    wm0 = {}
    wm1 = working.remember(wm0, "prior_denial_reason", "step therapy not documented")
    assert wm0 == {}
    assert working.recall(wm1, "prior_denial_reason") == "step therapy not documented"
    assert working.recall(wm1, "missing", default="?") == "?"


def test_search_working_matches_key_or_value():
    wm = working.remember(working.remember({}, "pt_weeks", 8), "dx", "M54.16")
    assert working.search_working(wm, "weeks") == [("pt_weeks", 8)]
    assert working.search_working(wm, "m54") == [("dx", "M54.16")]
```

```python
# tests/test_memory_tools.py
import pytest

from pa_copilot.memory.tools import build_memory_tools

NS = ("pa", "member", "M100001")


@pytest.mark.asyncio
async def test_manage_then_search_roundtrip(memory_store):
    manage, search = build_memory_tools(memory_store, NS)
    await manage.ainvoke({"content": "Member had lumbar MRI approved in 2025."})
    hits = await search.ainvoke({"query": "lumbar MRI history"})
    assert "lumbar" in str(hits).lower()


def test_manage_write_is_policy_governed(memory_store):
    manage, _ = build_memory_tools(memory_store, NS)
    manage.invoke({"content": "note governed by PolicyStore"})
    items = memory_store.search(NS, limit=10)
    assert items and all("importance" in i.value for i in items)
```

> `pytest-asyncio` is configured (`asyncio_default_fixture_loop_scope = "function"`). Use
> `@pytest.mark.asyncio`. If LangMem's tool only exposes a sync path in 0.0.30, drop the
> async test and keep `test_manage_write_is_policy_governed` (sync `.invoke`) — the
> policy-governance assertion is the load-bearing one.

- [ ] **Step 2: Run red**

Run: `python -m pytest tests/test_memory_tools.py tests/test_memory_working.py -v -m "not slow"`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `memory/working.py`**

```python
"""Tier-1 working memory: a tiny typed API over `state["working_memory"]["facts"]`
so a worker can recall a fact stated on an earlier turn (AC-06, short-term tier).
Persistence of `working_memory` across a pause/resume is the checkpointer's job
(PR5); this module is pure dict transforms."""

from __future__ import annotations

from typing import Any


def remember(working_memory: dict[str, Any] | None, key: str, value: Any) -> dict[str, Any]:
    wm = dict(working_memory or {})
    facts = dict(wm.get("facts") or {})
    facts[key] = value
    wm["facts"] = facts
    return wm


def recall(working_memory: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    return (working_memory or {}).get("facts", {}).get(key, default)


def search_working(working_memory: dict[str, Any] | None, substring: str) -> list[tuple[str, Any]]:
    q = substring.lower()
    facts = (working_memory or {}).get("facts", {})
    return [(k, v) for k, v in facts.items() if q in k.lower() or q in str(v).lower()]
```

- [ ] **Step 4: Implement `memory/tools.py`**

```python
"""LangMem `manage_memory` / `search_memory` bound to a PolicyStore. In PR5 these
bind to the `intake` (read member/provider history) and `decision_draft` (write
the determination) workers; PR4 builds and unit-tests them standalone."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langmem import create_manage_memory_tool, create_search_memory_tool

from pa_copilot.memory.store import PolicyStore


def build_memory_tools(
    store: PolicyStore,
    namespace: tuple[str, ...] | str,
    *,
    manage_instructions: str | None = None,
) -> tuple[BaseTool, BaseTool]:
    kw = {"instructions": manage_instructions} if manage_instructions else {}
    manage = create_manage_memory_tool(namespace=namespace, store=store, **kw)
    search = create_search_memory_tool(namespace=namespace, store=store)
    return manage, search
```

- [ ] **Step 5: Wire `memory/__init__.py` re-exports**

Add the imports from Task 2's note:

```python
from pa_copilot.memory.store import PolicyStore, open_memory_store
from pa_copilot.memory.tools import build_memory_tools

__all__ = ["PolicyStore", "open_memory_store", "build_memory_tools"]
```

- [ ] **Step 6: Run green + lint + fast suite**

Run: `ruff check src tests && python -m pytest -q -m "not slow"`
Expected: clean; all PASS pristine.

- [ ] **Step 7: Commit**

```bash
git add src/pa_copilot/memory/tools.py src/pa_copilot/memory/working.py \
        src/pa_copilot/memory/__init__.py tests/test_memory_tools.py tests/test_memory_working.py
git commit -m "feat(memory): LangMem tool factory + Tier-1 working-memory API"
```

---

### Task 6: AC-06 — tiered memory recall + committed trace

**Files:**
- Create: `scripts/memory_demo.py`
- Create: `traces/tiered_memory_recall.json`
- Test: `tests/test_ac06_tiered_memory.py` (create)

**Interfaces:**
- Consumes: `memory.working`, `memory.store.memory_store`, `memory.policy.search_ranked`,
  `tests/_fakes.py::FakeEmbedder`.
- Produces: `scripts/memory_demo.py::build_payload(store) -> dict` and `main()` (writes the
  trace); a byte-stable `traces/tiered_memory_recall.json` (no timestamps).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ac06_tiered_memory.py
"""AC-06: tiered memory. Tier 1 (working memory) recalls a fact stated on an
earlier simulated turn; Tier 2 (long-term semantic store) recalls a
prior-session record. Also guards the committed trace's byte-stability."""

import json
from pathlib import Path

from pa_copilot.memory import policy, working
from pa_copilot.memory.store import memory_store

TRACE = Path(__file__).resolve().parents[1] / "traces" / "tiered_memory_recall.json"
NS = ("pa", "member", "M100001")


def test_working_memory_recalls_a_fact_from_an_earlier_turn():
    wm = {}
    wm = working.remember(wm, "conservative_care_weeks", 8)          # turn 1
    wm = working.remember(wm, "chief_complaint", "radicular pain")   # turn 2
    assert working.recall(wm, "conservative_care_weeks") == 8        # turn 3
    assert working.search_working(wm, "radicular") == [("chief_complaint", "radicular pain")]


def test_long_term_store_recalls_prior_record(tmp_path, fake_embedder):
    with memory_store(tmp_path / "m.db", embedder=fake_embedder) as store:
        store.put(NS, "det-2025-11", {
            "content": "Prior auth for lumbar MRI 72148 approved 2025-11; 8 weeks PT documented.",
            "importance": "notable",
        })
        hits = policy.search_ranked(store, NS, "history of lumbar MRI prior authorization", limit=3)
        assert hits and hits[0].key == "det-2025-11"
        assert "AC-06"  # id marker


def test_committed_trace_is_byte_stable(tmp_path, fake_embedder):
    from scripts.memory_demo import build_payload
    with memory_store(tmp_path / "m.db", embedder=fake_embedder) as store:
        fresh = json.dumps(build_payload(store), indent=2, sort_keys=True) + "\n"
    assert fresh == TRACE.read_text(encoding="utf-8")
```

> `from scripts.memory_demo import build_payload` needs `scripts/` importable. `pyproject`
> `pythonpath = ["."]` puts the repo root on `sys.path`, and `scripts/` has no `__init__.py`
> — use `import importlib.util` to load it by path if the bare import fails, OR add
> `scripts/__init__.py` (check PR3: `scripts/` has none and `ingest_rag.py` is only run, not
> imported). Cleanest: load by path with a helper at the top of the test. Decide during
> implementation; the byte-stability contract is what matters.

- [ ] **Step 2: Run red**

Run: `python -m pytest tests/test_ac06_tiered_memory.py -v -m "not slow"`
Expected: FAIL (no `scripts.memory_demo`, no trace file).

- [ ] **Step 3: Implement `scripts/memory_demo.py`**

```python
"""In-repo producer of traces/tiered_memory_recall.json (AC-06 evidence).

    python scripts/memory_demo.py

Deterministic: the FakeEmbedder (no model download) + no timestamps in the
payload, so the committed bytes are stable across runs. tests/test_ac06_tiered_memory.py
regenerates and diffs."""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from pa_copilot.memory import policy, working  # noqa: E402
from pa_copilot.memory.store import memory_store  # noqa: E402

NS = ("pa", "member", "M100001")


def build_payload(store) -> dict:
    # Tier 1 — working memory across simulated turns
    wm = {}
    wm = working.remember(wm, "conservative_care_weeks", 8)
    wm = working.remember(wm, "chief_complaint", "radicular pain")
    turn3 = working.recall(wm, "conservative_care_weeks")

    # Tier 2 — long-term semantic store
    store.put(NS, "det-2025-11", {
        "content": "Prior auth for lumbar MRI 72148 approved 2025-11; 8 weeks PT documented.",
        "importance": "notable",
    })
    hits = policy.search_ranked(store, NS, "history of lumbar MRI prior authorization", limit=1)
    top = hits[0]

    return {
        "ac": "AC-06",
        "scenario": "MRI lumbar re-request for member M100001; prior determination recalled",
        "tier_1_working_memory": {
            "writes": [
                {"turn": 1, "key": "conservative_care_weeks", "value": 8},
                {"turn": 2, "key": "chief_complaint", "value": "radicular pain"},
            ],
            "turn_3_recall": {"key": "conservative_care_weeks", "recalled": turn3},
        },
        "tier_2_long_term": {
            "namespace": list(NS),
            "written_key": "det-2025-11",
            "query": "history of lumbar MRI prior authorization",
            "top_hit_key": top.key,
            "top_hit_content": top.value["content"],
            "top_hit_importance": top.value["importance"],
        },
        "embedder": "FakeEmbedder",
        "semantic_index": store.semantic_index_available,
    }


def main() -> None:
    out = _REPO_ROOT / "traces" / "tiered_memory_recall.json"
    with memory_store(_REPO_ROOT / ".pa_memory_demo.db", embedder=_fake()) as store:
        payload = build_payload(store)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    (_REPO_ROOT / ".pa_memory_demo.db").unlink(missing_ok=True)
    print(out.relative_to(_REPO_ROOT).as_posix())


def _fake():
    from tests._fakes import FakeEmbedder
    return FakeEmbedder()


if __name__ == "__main__":
    main()
```

> `.pa_memory_demo.db` is covered by `.gitignore`'s `*.sqlite3`? No — add `.pa_memory_demo.db*`
> to `.gitignore` in this task, or reuse a `tmp` path. Simplest: write to
> `_REPO_ROOT / ".pa_memory.db"` is wrong (production file). Add the ignore line.

- [ ] **Step 4: Generate the trace, run green**

Run: `python scripts/memory_demo.py && python -m pytest tests/test_ac06_tiered_memory.py -q -m "not slow"`
Expected: trace written; tests PASS. Run `python scripts/memory_demo.py` twice — `git status`
shows no diff on the second run (byte-stable).

- [ ] **Step 5: Lint + fast suite + commit**

```bash
git add scripts/memory_demo.py traces/tiered_memory_recall.json \
        tests/test_ac06_tiered_memory.py .gitignore
git commit -m "test(memory): AC-06 tiered recall + byte-stable trace + producer"
```

---

### Task 7: AC-08 — eviction / TTL sweep test

**Files:**
- Test: `tests/test_ac08_eviction.py` (create)
- (No new source — exercises Task 4's `policy.enforce_cap` / `policy.sweep_expired`.)

**Interfaces:**
- Consumes: `memory.store.memory_store`, `memory.policy` (`enforce_cap`, `sweep_expired`,
  `search_ranked`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ac08_eviction.py
"""AC-08: the memory eviction/importance policy. Overflow a namespace ->
lowest importance*recency non-critical item is evicted, `critical` is retained;
an expired item is gone after sweep_expired(). Runs on the non-semantic store
(index=None) so it is pure policy logic."""

import pytest

from pa_copilot.config import get_settings
from pa_copilot.memory import policy
from pa_copilot.memory.store import memory_store

NS = ("pa", "episodic")


@pytest.fixture
def small_episodic_cap(monkeypatch):
    s = get_settings()
    NP = type(s.memory.namespaces["episodic"])
    monkeypatch.setitem(s.memory.namespaces, "episodic", NP(ttl_days=90, cap=3))


def test_overflow_evicts_lowest_non_critical(small_episodic_cap, tmp_path):
    with memory_store(tmp_path / "e.db", semantic=False) as store:
        store.put(NS, "routine_a", {"content": "case summary A"})
        store.put(NS, "critical_denial", {"content": "denied - appeal filed", "importance": "critical"})
        store.put(NS, "routine_b", {"content": "case summary B"})
        store.put(NS, "routine_c", {"content": "case summary C"})  # 4 > 3
        keys = {i.key for i in store.search(NS, limit=50)}
        assert "critical_denial" in keys
        assert len(keys) == 3
        assert "routine_a" not in keys  # oldest routine evicted


def test_all_critical_overflow_is_not_evicted(small_episodic_cap, tmp_path):
    with memory_store(tmp_path / "e2.db", semantic=False) as store:
        for k in ("c1", "c2", "c3", "c4"):
            store.put(NS, k, {"content": k, "importance": "critical"})
        keys = {i.key for i in store.search(NS, limit=50)}
        assert keys == {"c1", "c2", "c3", "c4"}  # cap is soft against critical


def test_expired_item_gone_after_sweep(tmp_path):
    with memory_store(tmp_path / "e3.db", semantic=False) as store:
        store.put(NS, "fresh", {"content": "keep me"})
        store.put(NS, "stale", {"content": "expire me"})
        store.conn.execute(
            "UPDATE store SET expires_at = '2000-01-01 00:00:00' WHERE key = 'stale'"
        )
        assert policy.sweep_expired(store) == 1
        assert store.get(NS, "stale") is None
        assert store.get(NS, "fresh") is not None
```

- [ ] **Step 2: Run red → implement (nothing new; Task 4 covers it) → run green**

Run: `python -m pytest tests/test_ac08_eviction.py -v -m "not slow"`
Expected: PASS (if a case fails, fix `policy.enforce_cap` / the `PolicyStore.put` post-write
hook in `memory/store.py` — this test is the AC-08 spec).

- [ ] **Step 3: Lint + fast suite + commit**

```bash
git add tests/test_ac08_eviction.py
git commit -m "test(memory): AC-08 eviction — LRU cap + critical retention + TTL sweep"
```

---

### Task 8: AC-07 — cross-session persistence (two processes) + log + Makefile

**Files:**
- Create: `scripts/run_persistence_test.py`
- Create: `traces/memory_persistence.log`
- Test: `tests/test_memory_persistence.py` (create)
- Modify: `Makefile` (`persistence-test` target)

**Interfaces:**
- Consumes: `memory.store.open_memory_store`, `tests/_fakes.py::FakeEmbedder`.
- Produces: `scripts/run_persistence_test.py` with `write_session(db_path) -> None`,
  `read_session(db_path) -> dict`, `main() -> int` (spawns two `sys.executable` subprocesses,
  combines their stdout, writes `traces/memory_persistence.log` with volatile fields masked,
  exits non-zero on mismatch).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_persistence.py
"""AC-07: memory persists across sessions. Form 1 — in-process: session A writes,
all objects torn down, session B opens a FRESH store at the same path and recalls
by exact key (no LLM). Form 2 — two real `python` processes via the committed
script; its combined stdout is the committed traces/memory_persistence.log."""

import subprocess
import sys
from pathlib import Path

from pa_copilot.memory.store import memory_store

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "traces" / "memory_persistence.log"
NS = ("pa", "member", "M100001")
REC = {"content": "Determination: lumbar MRI 72148 APPROVED; cited PA-MRI-LUMBAR step-therapy.",
       "importance": "critical", "disposition": "approve"}


def test_in_process_teardown_then_fresh_store_recalls(tmp_path, fake_embedder):
    db = tmp_path / "persist.db"
    with memory_store(db, embedder=fake_embedder) as a:
        a.put(NS, "det-1", REC)
    # `a` and its connection are closed here.
    with memory_store(db, embedder=fake_embedder) as b:
        got = b.get(NS, "det-1")
    assert got is not None
    assert got.value["disposition"] == "approve"
    assert got.value["content"] == REC["content"]


def test_committed_log_matches_a_fresh_two_process_run(tmp_path):
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_persistence_test.py"),
         "--db", str(tmp_path / "x.db"), "--stdout-only"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout == LOG.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run red**

Run: `python -m pytest tests/test_memory_persistence.py -v -m "not slow"`
Expected: FAIL (no script, no log).

- [ ] **Step 3: Implement `scripts/run_persistence_test.py`**

```python
"""AC-07 cross-session persistence, two real processes.

    python scripts/run_persistence_test.py            # writes traces/memory_persistence.log
    python scripts/run_persistence_test.py --stdout-only   # print, don't write (test use)

Process 1 writes a determination record and exits. Process 2 is a brand-new
interpreter that opens a fresh store at the same path and recalls the record by
exact key — no LLM, exact string match. Combined stdout (volatile timestamps
masked as <ts>) is the committed evidence."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

NS = ("pa", "member", "M100001")
KEY = "det-cross-session"
REC = {
    "content": "Determination: lumbar MRI 72148 APPROVED; cited PA-MRI-LUMBAR step-therapy.",
    "importance": "critical",
    "disposition": "approve",
}
_TS = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?")


def _fake():
    from tests._fakes import FakeEmbedder
    return FakeEmbedder()


def write_session(db_path: str) -> None:
    from pa_copilot.memory.store import memory_store
    with memory_store(db_path, embedder=_fake()) as store:
        store.put(NS, KEY, REC)
        print(f"[session-A pid={_pid()}] wrote {list(NS)}/{KEY}: {json.dumps(REC, sort_keys=True)}")


def read_session(db_path: str) -> int:
    from pa_copilot.memory.store import memory_store
    with memory_store(db_path, embedder=_fake()) as store:
        item = store.get(NS, KEY)
    if item is None:
        print(f"[session-B pid={_pid()}] MISS — persistence FAILED")
        return 1
    ok = item.value == REC
    print(f"[session-B pid={_pid()}] recalled {list(NS)}/{KEY}: {json.dumps(item.value, sort_keys=True)}")
    print(f"[session-B pid={_pid()}] exact-match={ok}")
    return 0 if ok else 1


def _pid() -> str:
    return "<pid>"  # masked for byte-stability


def _run_child(db_path: str, mode: str) -> str:
    res = subprocess.run(
        [sys.executable, __file__, "--db", db_path, "--_child", mode],
        capture_output=True, text=True,
    )
    return _TS.sub("<ts>", res.stdout), res.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--stdout-only", action="store_true")
    ap.add_argument("--_child", choices=["write", "read"])
    args = ap.parse_args()

    if args._child == "write":
        return 0 if write_session(args.db) is None else 0
    if args._child == "read":
        return read_session(args.db)

    Path(args.db).unlink(missing_ok=True)
    a_out, a_rc = _run_child(args.db, "write")
    b_out, b_rc = _run_child(args.db, "read")
    combined = (
        "# AC-07 cross-session memory persistence — two python processes\n"
        f"{a_out}{b_out}"
        f"RESULT: {'PASS' if (a_rc == 0 and b_rc == 0) else 'FAIL'}\n"
    )
    if args.stdout_only:
        sys.stdout.write(combined)
    else:
        (_REPO_ROOT / "traces" / "memory_persistence.log").write_text(
            combined, encoding="utf-8", newline="\n"
        )
        Path(args.db).unlink(missing_ok=True)
        print("wrote traces/memory_persistence.log")
    return 0 if (a_rc == 0 and b_rc == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

> Watch the `write_session` return contract (it returns `None`); tidy the `main` branch.
> The child subprocesses import `tests._fakes` — that needs the repo root on `sys.path`,
> which the script's own `sys.path.insert` handles. If `tests` is not importable from a
> child cwd, add `scripts/_persist_fake.py` with a copy of the 12-line `FakeEmbedder` and
> import that instead (keep it crc32-based for cross-process stability).

- [ ] **Step 4: Generate the log, run green**

Run: `python scripts/run_persistence_test.py && python -m pytest tests/test_memory_persistence.py -q -m "not slow"`
Expected: log written; both tests PASS. Regenerate twice → `git status` clean (byte-stable).

- [ ] **Step 5: Retarget the Makefile**

```make
persistence-test:
	python scripts/run_persistence_test.py
```

(was `pac persistence-test` — `pac` lands in PR7. Leave a `# pac persistence-test once PR7
wires the CLI` comment.)

- [ ] **Step 6: Lint + fast suite + commit**

```bash
git add scripts/run_persistence_test.py traces/memory_persistence.log \
        tests/test_memory_persistence.py Makefile
git commit -m "test(memory): AC-07 cross-session persistence — 2-process driver + log"
```

---

### Task 9: Docs — memory-policy.md, design/spec reconcile, ledgers

**Files:**
- Create: `docs/memory-policy.md`
- Modify: `docs/design.md` (§1 table row, §6.3 TTL note)
- Modify: `specs/acceptance-criteria.md` (AC-06/07/08 rows → `done`)
- Modify: `docs/rubric-coverage.md` (Memory Systems row)
- Modify: `docs/BUILD_LOG.md` (add the PR4 section — see below)

**Interfaces:** none (docs only). `tests/test_ac_traceability.py` and
`tests/test_nfr06_docs_present.py` must stay green.

- [ ] **Step 1: Write `docs/memory-policy.md`**

Cover, with the real symbol names from this PR:
- **Tiers** — table: Tier-1 `state["working_memory"]["facts"]` via `memory/working.py`
  (checkpointer-persisted in PR5); Tier-2 `PolicyStore` (`.pa_memory.db`) + optional
  sqlite-vec index, namespaces `("pa","member",<id>)` / `("pa","provider",<npi>)` /
  `("pa","policy_notes")` / `("pa","episodic")`.
- **TTL** — per-namespace `ttl_days` from `config/memory.yaml` → minutes, injected on every
  `PolicyStore.put` (native `TTLConfig` only has one global `default_ttl`, so the subclass
  does per-item TTL). `policy_notes` never expires. `refresh_on_read=True`.
  `policy.sweep_expired(store)` on startup (PR7 CLI) / `store.start_ttl_sweeper()` in PR5.
- **Importance** — `{routine:1.0, notable:1.5, critical:3.0}`; default `routine`; denials /
  appeals write `critical`. Stamped into `value["importance"]` on write.
- **LRU cap** — per-namespace `cap`; on overflow, `policy.enforce_cap` deletes the lowest
  `score × importance_weight × recency_decay`, **never `critical`** (cap is soft against
  critical — documented limitation).
- **Ranked search** — `policy.search_ranked` = `store.search` (semantic when the index is
  up, recency-only otherwise) re-ranked by the same combined key.
- **sqlite-vec fallback** — if the extension will not load, `open_memory_store` rebuilds
  with `index=None`; semantic recall degrades to importance×recency ranking; AC-06/07/08
  still hold. `store.semantic_error` records why.
- **Cross-session (AC-07)** — `scripts/run_persistence_test.py`, two processes, exact-key
  match, `traces/memory_persistence.log`.
- A "Good-to-Have / PR8" line: `create_memory_store_manager` background importance-weighted
  extraction.

- [ ] **Step 2: `docs/design.md` edits**

- §1 table, "Long-term memory (AC-06/07/08)" row Notes: append
  "; `PolicyStore` subclass injects per-namespace TTL + importance on every write".
- §6.3 TTL bullet: add a sentence —
  "`TTLConfig` carries only a single global `default_ttl`, so per-namespace expiry
  (`episodic` 90d, `member`/`provider` 365d, `policy_notes` none) is applied by
  `PolicyStore.put` passing a per-item `ttl` computed by `policy.ttl_minutes_for`."

- [ ] **Step 3: `specs/acceptance-criteria.md`**

AC-06 → `done` — "PR4: `memory/working.py` Tier-1 + `PolicyStore` Tier-2 semantic recall;
`tests/test_ac06_tiered_memory.py`; `traces/tiered_memory_recall.json`".
AC-07 → `done` — "PR4: in-process teardown+rebuild + 2-process
`scripts/run_persistence_test.py`; `tests/test_memory_persistence.py`;
`traces/memory_persistence.log`".
AC-08 → `done` — "PR4: `memory/policy.py` TTL + importance + LRU cap; `docs/memory-policy.md`;
`tests/test_ac08_eviction.py`".

- [ ] **Step 4: `docs/rubric-coverage.md`** — Memory Systems row → PR4 / done, pointing at
`docs/memory-policy.md` + the three AC tests.

- [ ] **Step 5: `docs/BUILD_LOG.md`** — replace the "NEXT: PR4" section with a "PR4 — Memory
Subsystem (merged `<hash>`)" summary (leave the hash as `<pending merge>` — the controller
fills it at merge) and a "NEXT: PR5" stub carrying design §3 scope + the PR3/PR2 parked
items (`CorporaUnavailable` widen in `medical_necessity`; env-snapshot now done). Follow the
PR1/PR2/PR3 section shape.

- [ ] **Step 6: Verify meta-tests + full suite**

Run: `python -m pytest -q -m "not slow" && ruff check src tests`
Expected: `test_ac_traceability.py`, `test_nfr06_docs_present.py` green; all PASS pristine.

- [ ] **Step 7: Commit**

```bash
git add docs/memory-policy.md docs/design.md specs/acceptance-criteria.md \
        docs/rubric-coverage.md docs/BUILD_LOG.md
git commit -m "docs(memory): memory-policy.md + design/spec reconcile + AC-06/07/08 done"
```

---

### Task 10: Whole-branch verification (controller-run)

**Files:** none (verification only).

- [ ] **Step 1: Full suite, both markers**

Run: `python -m pytest -q -m "not slow"` then `python -m pytest -q -m slow`
Expected: fast — all PASS pristine (~110 tests). slow — `test_local_embeddings_real_bge_dims`
PASS (real bge-small, 384). No failures; no warnings.

- [ ] **Step 2: Lint**

Run: `ruff check src tests`
Expected: clean.

- [ ] **Step 3: Evidence byte-stability**

Run: `python scripts/memory_demo.py && python scripts/run_persistence_test.py && git status --porcelain`
Expected: no modifications (both traces regenerate identically).

- [ ] **Step 4: Import hygiene**

Run: `python -c "import pa_copilot.memory; import pa_copilot.memory.store; import pa_copilot.memory.tools"`
from a directory **outside** the repo root (e.g. `cd / && python -c ...` with `PYTHONPATH`
set to `<repo>/src`).
Expected: 0 bytes stderr, exit 0 — the memory package must not depend on repo-root-only
imports (the PR3 C1 lesson).

- [ ] **Step 5: No production db files created**

Run: `git status --porcelain` and `ls .pa_memory.db*` — the suite and scripts must only
write to `tmp_path` / gitignored demo paths.

- [ ] **Step 6: First-parent history check**

Run: `git log --first-parent --oneline -3` on `feat/memory` — 9 task commits since
`3de3eb5`. Then hand off to `superpowers:finishing-a-development-branch` for the `--no-ff`
merge (the controller fills the BUILD_LOG merge hash post-merge).

---

## Self-Review

**Spec coverage (`docs/design.md` §6):**
- §6.1 tiered (short-term working + long-term semantic) → Task 5 (`working.py`) + Task 4
  (`PolicyStore`); namespaces → `policy.namespace_policy` (Task 2). ✓
- §6.1 LangMem `manage_memory` / `search_memory` → Task 5 (`tools.py`). ✓
- §6.2 cross-session persistence, both forms (in-process + 2-process script) → Task 8. ✓
- §6.3 TTL (native `TTLConfig` + per-item injection), importance weighting, LRU cap,
  `search_ranked` = semantic × importance × recency → Tasks 2 + 4; `sweep_ttl` via
  `policy.sweep_expired` → Task 4/7. ✓
- §5.1 "Write" strategy (working scratch → `state["working_memory"]`; durable → `SqliteStore`
  via LangMem) → Tasks 4, 5. ✓
- `docs/memory-policy.md` → Task 9. ✓
- AC-06 test + `traces/tiered_memory_recall.json` → Task 6. ✓
- AC-07 test + script + `traces/memory_persistence.log` → Task 8. ✓
- AC-08 test + doc → Tasks 7, 9. ✓
- PR2/PR3 parked env-snapshot fixture → Task 1. ✓
- sqlite-vec fallback (design §11 risk) → Task 4. ✓

**Placeholder scan:** the three "decide during implementation" notes (scripts import
mechanism in Tasks 6/8; the async-tool test in Task 5; a possible narrow `filterwarnings`
entry in Task 4) each give a concrete default and a concrete fallback — they are not open
questions. No "TBD"/"add error handling"/bare "write tests".

**Type consistency:**
- `namespace` is always a `tuple[str, ...]`; `PolicyStore` calls `tuple(namespace)` before
  passing to `policy.*` (LangMem may hand a list). ✓
- `MemoryConfig.semantic_fields` is a `tuple`; `open_memory_store` does `list(...)` for the
  index `fields`. ✓
- `policy.ttl_minutes_for` / `importance_of` / `rank_key` signatures identical between Task 2
  (definition) and Task 4 (`PolicyStore._prep`, `enforce_cap`) and Task 6/7 (tests). ✓
- `open_memory_store` / `memory_store` share one keyword signature (`path, *, embedder,
  settings, semantic`) across Tasks 4, 6, 8. ✓
- `build_memory_tools(store, namespace, *, manage_instructions=None)` — Task 5 definition
  matches the Task 5 tests. ✓
- Trace producers both write `json.dumps(..., indent=2, sort_keys=True) + "\n"` /
  masked-text with `encoding="utf-8", newline="\n"` — matches the byte-stability constraint
  and the Task 6/8 guard tests. ✓
