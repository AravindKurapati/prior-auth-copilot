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
    """Regression for Task 7 review finding (Ruling R37): `created_at`/`updated_at`
    are stamped via SQL `CURRENT_TIMESTAMP`, which only has 1-second granularity, so
    items written back-to-back in a test tie on `recency_decay` (~1.0 for all of
    them). A tie means Python's stable sort would settle overflow on whichever item
    happens to sort first for OTHER reasons (e.g. insertion order) even if recency
    were a no-op — that would make this test pass for the wrong reason.

    To force a genuine recency signal: write two fresh routine items first, then
    the item that should be evicted, then directly backdate ONLY that item's
    timestamps (same raw-SQL-UPDATE technique as test_expired_item_gone_after_sweep
    below), and only then push the namespace over cap. `routine_old` is the third
    item written -- not the first -- so if `recency_decay` were broken and eviction
    fell back to insertion-order tie-breaking, `routine_a` (written first) would be
    evicted instead. The test can only pass as written if recency is genuinely
    driving the ranking.
    """
    with memory_store(tmp_path / "e.db", semantic=False) as store:
        store.put(NS, "routine_a", {"content": "case summary A"})
        store.put(NS, "routine_b", {"content": "case summary B"})
        store.put(NS, "routine_old", {"content": "case summary OLD"})
        old_ts = "2000-01-01 00:00:00"
        store.conn.execute(
            "UPDATE store SET created_at = ?, updated_at = ? WHERE key = 'routine_old'",
            (old_ts, old_ts),
        )
        store.put(
            NS,
            "critical_denial",
            {"content": "denied - appeal filed", "importance": "critical"},
        )  # 4 > 3, triggers eviction
        keys = {i.key for i in store.search(NS, limit=50)}
        assert "critical_denial" in keys
        assert len(keys) == 3
        assert "routine_old" not in keys  # genuinely oldest (backdated) item evicted
        assert "routine_a" in keys  # not evicted, despite being inserted first
        assert "routine_b" in keys


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
