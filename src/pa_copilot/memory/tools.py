"""LangMem `manage_memory` / `search_memory` bound to a PolicyStore. In PR5 these
bind to the `intake` (read member/provider history) and `decision_draft` (write
the determination) workers; PR4 builds and unit-tests them standalone."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langmem import create_manage_memory_tool, create_search_memory_tool

from pa_copilot.memory.store import PolicyStore


def build_memory_tools(
    store: PolicyStore,
    namespace: tuple[str, ...] | str,
    *,
    manage_instructions: str | None = None,
) -> tuple[BaseTool, BaseTool]:
    kw = {"instructions": manage_instructions} if manage_instructions else {}
    manage = create_manage_memory_tool(namespace=namespace, store=store, **kw)
    search = create_search_memory_tool(namespace=namespace, store=store)
    return manage, search
