"""PolicyStore write interception (AC-08), LRU cap, and the sqlite-vec fallback."""

import sqlite3
import time

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


def test_enforce_cap_scan_does_not_refresh_other_items_ttl(memory_store):
    """Finding I1 (Ruling R35) regression: enforce_cap's own internal
    full-namespace listing (used only to check the cap) must not count as a
    "read" of every other item in the namespace -- only a real query-serving
    search (search_ranked) is allowed to refresh TTL. Writing item B to the
    same TTL-bearing namespace must not push out item A's expires_at."""
    memory_store.put(NS_MEMBER, "a", {"content": "first item in the namespace"})
    row_before = memory_store.conn.execute(
        "SELECT expires_at FROM store WHERE key = ?", ("a",)
    ).fetchone()
    assert row_before[0] is not None

    # Cross a wall-clock second boundary so that, if the bug were still present,
    # the refreshed expires_at (CURRENT_TIMESTAMP + ttl_minutes, second
    # resolution) would provably differ from the pre-write value -- a same-
    # second write/write pair could coincidentally look unchanged either way.
    time.sleep(1.1)

    # A second, unrelated write into the SAME namespace triggers PolicyStore's
    # post-put enforce_cap housekeeping scan. Before the fix this scan's
    # store.search(...) refreshed TTL on every item in the namespace,
    # including "a", even though "a" was never itself read or written again.
    memory_store.put(NS_MEMBER, "b", {"content": "second item in the namespace"})

    row_after = memory_store.conn.execute(
        "SELECT expires_at FROM store WHERE key = ?", ("a",)
    ).fetchone()
    assert row_after[0] == row_before[0]


def test_open_memory_store_closes_both_conns_when_both_setup_attempts_fail(
    tmp_path, fake_embedder, monkeypatch
):
    """Finding I2 (Ruling R36) regression: if the primary setup() fails AND the
    non-semantic fallback's setup() also fails, both sqlite3 connections must
    be closed before the exception propagates -- neither may be leaked."""
    import pa_copilot.memory.store as store_mod

    created: list[PolicyStore] = []
    original_build = store_mod._build

    def spying_build(path, index, settings):
        built = original_build(path, index, settings)
        created.append(built)
        return built

    monkeypatch.setattr(store_mod, "_build", spying_build)
    monkeypatch.setattr(
        PolicyStore,
        "setup",
        lambda self: (_ for _ in ()).throw(RuntimeError("setup boom")),
    )

    with pytest.raises(RuntimeError, match="setup boom"):
        open_memory_store(tmp_path / "double_fail.db", embedder=fake_embedder)

    # Both the primary attempt (semantic index configured) and the fallback
    # attempt (index=None) must have run and been closed.
    assert len(created) == 2
    for built in created:
        with pytest.raises(sqlite3.ProgrammingError):
            built.conn.execute("SELECT 1")


def test_enforce_cap_and_search_ranked_reject_prefix_leak_across_namespaces(
    tmp_path, fake_embedder, monkeypatch
):
    """Final whole-branch review, Finding I1: SqliteStore.search matches
    namespaces by a raw SQL prefix with no trailing separator, so the
    dot-joined string for ("pa","member","M1") is a literal string-prefix of
    ("pa","member","M10")'s. `enforce_cap` and `search_ranked` must filter
    the raw search results to an EXACT namespace match before doing anything
    else with them, or M1's cap accounting / ranked search silently absorbs
    M10's records -- a data-isolation bug, not a ranking wobble."""
    s = get_settings()
    monkeypatch.setitem(
        s.memory.namespaces, "member",
        type(s.memory.namespaces["member"])(ttl_days=365, cap=2),
    )
    ns_m1 = ("pa", "member", "M1")
    ns_m10 = ("pa", "member", "M10")
    with memory_store(tmp_path / "prefix.db", embedder=fake_embedder) as store:
        # M10 sits at its own cap -- two real items, self-contained.
        store.put(ns_m10, "x", {"content": "m10 item x"})
        store.put(ns_m10, "y", {"content": "m10 item y"})

        # One M1 write is well under M1's own cap (1 < 2). Because M1's
        # dot-joined namespace ("pa.member.M1") is a literal string-prefix of
        # M10's ("pa.member.M10"), a buggy enforce_cap would see M10's two
        # items too and compute a phantom overflow that should never exist.
        store.put(ns_m1, "a", {"content": "m1 item a"})

        # (a) M10's items must never be evicted or counted against M1's cap.
        evicted = policy.enforce_cap(store, ns_m1, mem=s.memory)
        assert evicted == []
        assert store.get(ns_m1, "a") is not None
        assert store.get(ns_m10, "x") is not None
        assert store.get(ns_m10, "y") is not None

        # ...and vice versa: M1's item must never be evicted or counted
        # against M10's cap.
        evicted_m10 = policy.enforce_cap(store, ns_m10, mem=s.memory)
        assert evicted_m10 == []

        # (b) search_ranked on M1 must return ONLY M1's own record, never
        # anything from M10 (and vice versa).
        ranked_m1 = policy.search_ranked(store, ns_m1, None, limit=50)
        assert {i.key for i in ranked_m1} == {"a"}
        assert all(tuple(i.namespace) == ns_m1 for i in ranked_m1)

        ranked_m10 = policy.search_ranked(store, ns_m10, None, limit=50)
        assert {i.key for i in ranked_m10} == {"x", "y"}
        assert all(tuple(i.namespace) == ns_m10 for i in ranked_m10)
