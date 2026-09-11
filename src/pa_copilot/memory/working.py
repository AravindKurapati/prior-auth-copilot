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
