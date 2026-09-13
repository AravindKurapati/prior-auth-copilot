"""Background, importance-weighted memory consolidation (PR8 good-to-have —
design.md §6.1 names this exactly: "Good-to-Have: create_memory_store_manager
for background importance-weighted extraction").

Runs OUTSIDE the synchronous agent loop — called once a case reaches a
terminal state (FINISH or paused for human_review; see cli.py's submit/resume),
never from inside a graph node — so it touches no graph topology, routing, or
already-tested worker.

Unlike decision_draft.py's direct ``PolicyStore.put`` on deny (a hard rule:
"this exact fact, at this exact importance"), this manager reads the whole
case transcript and lets an LLM decide what is worth remembering long-term at
all, at what importance, and whether it supersedes an existing memory —
consolidating rather than only ever appending.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langmem import create_memory_store_manager
from langmem.knowledge.extraction import MemoryStoreManager

from pa_copilot.agents._react import get_lite_agent_model
from pa_copilot.memory.store import PolicyStore

#: design.md §6.1's fixed episodic namespace — case summaries live here,
#: separate from the per-member/per-provider namespaces intake/decision_draft
#: read and write directly.
EPISODIC_NAMESPACE: tuple[str, ...] = ("pa", "episodic")

_INSTRUCTIONS = (
    "You are the background memory consolidator for a prior-authorization "
    "copilot. Given the full case transcript, decide what is worth "
    "remembering long-term about this member/provider/policy pattern, and "
    "tag every memory's `importance` field as one of routine, notable, or "
    "critical (design.md §6.3's policy):\n"
    "- critical: any denial or appeal, or anything a future reviewer MUST see\n"
    "- notable: a pattern worth surfacing but not safety-critical (e.g. a "
    "recurring missing-field issue with a provider)\n"
    "- routine: everything else worth keeping at all\n"
    "Summarize rather than quoting raw clinical narrative verbatim. Skip "
    "anything already captured near-verbatim in an existing memory."
)


def build_case_memory_manager(store: PolicyStore, *, model: Any = None) -> MemoryStoreManager:
    """One manager bound to ``store``'s episodic namespace.

    ``model`` defaults to the lite model — reflection.py's degrade path
    already uses "cheap model for a cheap task"; case consolidation is not
    the case-critical path, so the same tradeoff applies here.
    """
    return create_memory_store_manager(
        model or get_lite_agent_model(),
        namespace=EPISODIC_NAMESPACE,
        store=store,
        instructions=_INSTRUCTIONS,
        enable_deletes=True,
    )


async def enrich_case_memory(
    manager: MemoryStoreManager, messages: list[BaseMessage]
) -> list[Any]:
    """Run ``manager`` once over a finished case's message history.

    Call this AFTER a case reaches FINISH or pauses for human_review — never
    from inside a graph node. Returns whatever the manager returns (the
    updated/inserted memory objects); callers that don't need the detail can
    ignore it.
    """
    return await manager.ainvoke({"messages": messages})
