"""PolicyStore write interception (AC-08), LRU cap, and the sqlite-vec fallback."""

from pa_copilot.config import get_settings
from pa_copilot.memory import policy
from pa_copilot.memory.store import memory_store

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
