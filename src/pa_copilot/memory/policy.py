"""Memory policy: per-namespace TTL, importance weighting, recency decay, and the
combined rank key used for LRU-cap eviction and ranked search (design.md §6.3).

Pure functions — a store handle is passed in where one is needed (Task 4). The
memory store (store.py) is what actually calls `ttl_minutes_for` / `importance_of`
on every write."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from pa_copilot.config import MemoryConfig, NamespacePolicy


class _Ranked(Protocol):
    value: dict[str, Any]
    updated_at: Any
    created_at: Any


def namespace_policy(
    namespace: tuple[str, ...], mem: MemoryConfig
) -> NamespacePolicy | None:
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


# --- Store-coupled policy ops (Task 4) -------------------------------------
# These take a live store handle. The pure scalars above stay store-free; the
# functions below drive ranked retrieval and LRU-cap eviction against a store
# whose writes PolicyStore already governs (design.md §6.3).


def _settings_mem() -> MemoryConfig:
    # Deferred import: `pa_copilot.config` is fine to import at module load, but
    # keeping it local mirrors the store/policy split and avoids any import cycle
    # if config ever grows a memory dependency.
    from pa_copilot.config import get_settings

    return get_settings().memory


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def search_ranked(
    store: Any,
    namespace: tuple[str, ...],
    query: str | None,
    *,
    limit: int | None = None,
    mem: MemoryConfig | None = None,
    now: datetime | None = None,
    pool: int = 50,
) -> list[Any]:
    """Fetch a generous pool from the store, then re-order it by `rank_key`
    (importance * recency * semantic score) descending. When the store has no
    vector index, `query` is dropped and the pool is recency/importance-ranked."""
    mem = mem or _settings_mem()
    when = _now(now)
    use_query = query if getattr(store, "semantic_index_available", True) else None
    raw = list(store.search(namespace, query=use_query, limit=max(pool, limit or 0)))
    raw.sort(key=lambda it: rank_key(it, mem, now=when), reverse=True)
    return raw[:limit] if limit else raw


def enforce_cap(
    store: Any,
    namespace: tuple[str, ...],
    *,
    mem: MemoryConfig | None = None,
    now: datetime | None = None,
) -> list[str]:
    """Delete the lowest-`rank_key` non-`critical` items until the namespace is
    at its cap. The cap is SOFT against `critical`: if only critical items remain
    over cap, they are left in place. Returns the evicted keys."""
    mem = mem or _settings_mem()
    pol = namespace_policy(namespace, mem)
    if pol is None or not pol.cap:
        return []
    # This listing is internal housekeeping, not a real read of any one item —
    # it must not refresh TTL on every other item in the namespace (that would
    # turn "N days since an item was last written/touched" into "N days since
    # anything in the namespace was written", defeating per-item TTL policy).
    items = list(store.search(namespace, limit=10_000, refresh_ttl=False))
    overflow = len(items) - pol.cap
    if overflow <= 0:
        return []
    when = _now(now)
    items.sort(key=lambda it: rank_key(it, mem, now=when))  # worst first
    evicted: list[str] = []
    for it in items:
        if overflow <= 0:
            break
        if importance_of(getattr(it, "value", {}) or {}, mem) == "critical":
            continue
        store.delete(namespace, it.key)
        evicted.append(it.key)
        overflow -= 1
    return evicted


def sweep_expired(store: Any) -> int:
    """Opportunistic TTL sweep — deletes rows whose `expires_at` has passed."""
    return store.sweep_ttl()
