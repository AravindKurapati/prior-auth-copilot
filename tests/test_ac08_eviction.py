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
