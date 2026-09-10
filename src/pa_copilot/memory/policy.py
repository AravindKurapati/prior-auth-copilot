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
