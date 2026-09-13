"""design.md §8.2: left = pick/paste a request; center = live routing trail +
structured artifacts filling in; right = memory panel. Business logic stays in
pa_copilot -- this module renders what make_graph/tracing already produce.

Testable surface: the module-level pure functions below (_route_rows,
_artifact_rows, _memory_panel_rows) are unit-tested directly; the `st.*` calls in
main() are exercised only by `pac demo` launching the real Streamlit process.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import streamlit as st  # noqa: E402


def _route_rows(route_history: list[Any]) -> list[dict]:
    """RouteStep list -> plain dicts for st.dataframe/st.table -- pure, no Streamlit call."""
    return [
        {"from": step.from_node, "to": step.to_node, "reason": step.reason}
        for step in route_history
    ]


def _artifact_rows(state: dict) -> dict:
    """The four structured artifacts, model_dump()'d where present, else None --
    what the center panel fills in as they arrive."""
    out = {}
    for key in ("request", "benefit", "necessity", "decision"):
        val = state.get(key)
        out[key] = val.model_dump() if val is not None else None
    return out


def _memory_panel_rows(working_memory: dict | None, long_term_hits: list[dict] | None) -> dict:
    """Right panel: working memory keys/values + any long-term store search hits
    used this run (caller supplies long_term_hits; this module doesn't call the
    store directly)."""
    return {"working_memory": dict(working_memory or {}), "long_term_hits": list(long_term_hits or [])}


async def _run_one_case(raw_provider_text: str, case_id: str, session_id: str, member_id: str) -> dict:
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from pa_copilot.cli import _build_graph_for_case
    from pa_copilot.config import get_settings
    from pa_copilot.memory.store import memory_store
    from pa_copilot.state import new_case_state

    settings = get_settings()
    conn = await aiosqlite.connect(settings.state_db)
    try:
        checkpointer = AsyncSqliteSaver(conn)
        with memory_store(settings.memory_db, settings=settings) as store:
            graph = await _build_graph_for_case(store, checkpointer)
            state = new_case_state(case_id, session_id, member_id, raw_provider_text)
            thread = {"configurable": {"thread_id": case_id}}
            return await graph.ainvoke(state, config=thread)
    finally:
        await conn.close()


def main() -> None:
    import asyncio
    import json
    import uuid

    st.title("Prior-Authorization Copilot")
    left, center, right = st.columns([1, 2, 1])

    sample_dir = _REPO_ROOT / "data" / "samples"
    with left:
        choice = st.selectbox("Sample request", sorted(p.name for p in sample_dir.glob("*.json")))
        pasted = st.text_area("...or paste a raw provider note", "")
        run_clicked = st.button("Run")
    with center:
        st.subheader("Routing trail")
        trail_placeholder = st.empty()
        st.subheader("Artifacts")
        artifact_placeholder = st.empty()
    with right:
        st.subheader("Memory")
        memory_placeholder = st.empty()

    if not run_clicked:
        return

    if pasted.strip():
        raw_provider_text = pasted
        case_id = f"case-{uuid.uuid4().hex[:8]}"
        session_id = f"sess-{uuid.uuid4().hex[:8]}"
        member_id = "unknown"
    else:
        sample = json.loads((sample_dir / choice).read_text(encoding="utf-8"))
        raw_provider_text = sample["raw_provider_text"]
        case_id = sample["case_id"]
        session_id = sample["session_id"]
        member_id = sample["member_id"]

    try:
        result = asyncio.run(_run_one_case(raw_provider_text, case_id, session_id, member_id))
    except Exception as exc:  # noqa: BLE001 -- surface in the UI, don't crash the app
        st.error(str(exc))
        return

    trail_placeholder.table(_route_rows(result.get("route_history") or []))
    artifact_placeholder.json(_artifact_rows(result))
    memory_placeholder.json(_memory_panel_rows(result.get("working_memory"), []))


if __name__ == "__main__":
    main()
