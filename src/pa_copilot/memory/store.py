"""Long-term semantic memory: a SqliteStore subclass that enforces the design.md
§6.3 policy on every write, plus a factory that degrades to a non-semantic store
when sqlite-vec will not load."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import Iterable, Iterator

from langgraph.store.base import NOT_PROVIDED, Op, Result, TTLConfig
from langgraph.store.sqlite import SqliteStore

from pa_copilot.config import Settings, get_settings
from pa_copilot.memory import policy
from pa_copilot.memory.embeddings import as_embeddings, embedding_dims


class PolicyStore(SqliteStore):
    """Every put/aput gets the namespace's TTL and a default importance stamped
    on, then the namespace's LRU cap enforced. LangMem's manage_memory tool calls
    ``store.put``, so its writes are governed too. NOTE: a raw ``store.batch([...])``
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
        mem = self._mem
        if "importance" not in (value or {}):
            value = {**(value or {}), "importance": policy.importance_of(value or {}, mem)}
        if ttl is NOT_PROVIDED:
            ttl = policy.ttl_minutes_for(tuple(namespace), mem)
        return value, ttl

    def put(self, namespace, key, value, index=None, *, ttl=NOT_PROVIDED) -> None:
        value, ttl = self._prep(namespace, value, ttl)
        super().put(namespace, key, value, index=index, ttl=ttl)
        policy.enforce_cap(self, tuple(namespace), mem=self._mem)

    async def aput(self, namespace, key, value, index=None, *, ttl=NOT_PROVIDED) -> None:
        value, ttl = self._prep(namespace, value, ttl)
        await super().aput(namespace, key, value, index=index, ttl=ttl)
        policy.enforce_cap(self, tuple(namespace), mem=self._mem)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """``SqliteStore.abatch`` (langgraph's own override, not ours)
        unconditionally raises ``NotImplementedError`` for every async batch
        call -- "The SqliteStore does not support async methods. Consider
        using AsyncSqliteStore instead." That breaks every async caller that
        goes through the generic ``BaseStore`` async surface: LangMem's
        ``search_memory`` tool's async variant calls ``store.asearch(...)``,
        which is ``BaseStore.asearch`` batching a single ``SearchOp`` through
        ``abatch`` (verified by reading the installed
        ``langgraph.store.base.BaseStore.asearch``/``aput`` source — both are
        thin wrappers that build one ``Op`` and delegate to
        ``self.abatch([op])``). Any worker whose ReAct loop runs via
        ``.ainvoke()`` (langgraph's ``ToolNode`` always dispatches a bound
        tool's async path when the graph itself is invoked asynchronously)
        therefore hits this immediately -- 100% failure, not a corner case.

        ``SqliteStore.batch`` (sync) has a real, working implementation, and
        sqlite access here is a fast local file operation, not a network
        call, so delegating the async batch to a thread is a safe, minimal
        fix: no event loop is blocked for any meaningful duration, and no
        server-side behavior changes -- only the entry point does.

        This does NOT bypass the TTL/importance/cap-enforcement policy this
        class stamps on writes: ``put``/``aput`` above call ``self._prep(...)``
        BEFORE ever reaching ``batch``/``abatch``, so a value routed through
        here (e.g. via ``aput`` -> ``BaseStore.aput`` -> ``self.abatch([PutOp
        (already-prepped value)])``) already carries the stamped
        importance/ttl; cap enforcement then runs in ``put``/``aput`` itself,
        after the (a)batch call returns. A *read-only* search
        (``search``/``asearch`` -> ``SearchOp`` -> batch/abatch) never touches
        ``put``/``aput``/``_prep`` at all, so there is nothing for this
        override to bypass on that path either. The one pre-existing caveat
        (documented on the class above) still applies unchanged: a raw
        ``store.batch([...])``/``store.abatch([...])`` call bypasses policy
        injection because it skips ``put``/``aput`` entirely -- first-party
        code must keep using ``put``/``aput``, never call `(a)batch` directly
        with a `PutOp`.
        """
        return await asyncio.to_thread(self.batch, list(ops))


def _build(path: str, index, settings: Settings) -> PolicyStore:
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    return PolicyStore(
        conn, settings=settings, index=index, ttl=TTLConfig(refresh_on_read=True)
    )


def open_memory_store(
    path: str | None = None,
    *,
    embedder: object | None = None,
    settings: Settings | None = None,
    semantic: bool = True,
) -> PolicyStore:
    s = settings or get_settings()
    db_path = str(path or s.memory_db)
    index = None
    if semantic and embedder is not None:
        index = {
            "dims": embedding_dims(embedder),
            "embed": as_embeddings(embedder),
            "fields": list(s.memory.semantic_fields),
        }
    store = _build(db_path, index, s)
    try:
        store.setup()
    except Exception as exc:  # noqa: BLE001 -- sqlite-vec extension load failure
        store.conn.close()
        if index is None:
            raise
        store = _build(db_path, None, s)
        store.semantic_error = repr(exc)
        try:
            store.setup()
        except Exception:
            store.conn.close()
            raise
    return store


@contextlib.contextmanager
def memory_store(
    path: str | None = None,
    *,
    embedder: object | None = None,
    settings: Settings | None = None,
    semantic: bool = True,
) -> Iterator[PolicyStore]:
    store = open_memory_store(
        path, embedder=embedder, settings=settings, semantic=semantic
    )
    try:
        yield store
    finally:
        store.conn.close()
