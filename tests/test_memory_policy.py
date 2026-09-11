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
    assert policy.recency_decay(
        older, now=NOW, half_life_days=mem.recency_half_life_days
    ) == pytest.approx(0.5, rel=1e-6)
    assert policy.recency_decay(NOW, now=NOW, half_life_days=30) == pytest.approx(1.0)
    assert policy.recency_decay("not-a-date", now=NOW, half_life_days=30) == 1.0


def test_rank_key_combines_factors(mem):
    class Item:
        value = {"importance": "critical"}
        updated_at = NOW
        created_at = NOW
        score = 0.5

    assert policy.rank_key(Item(), mem, now=NOW) == pytest.approx(0.5 * 3.0 * 1.0)
