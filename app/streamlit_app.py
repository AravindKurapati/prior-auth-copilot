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


def _long_term_hits(store: Any, member_id: str, *, limit: int = 10) -> list[dict]:
    """PR8 good-to-have: the member's long-term memories, importance/recency
    ranked via memory/policy.py's search_ranked -- what the memory panel
    previously always rendered as an empty list. Pure aside from the store
    call (mirrors policy.search_ranked's own store-coupled-but-otherwise-pure
    shape)."""
    from pa_copilot.memory import policy  # noqa: PLC0415

    items = policy.search_ranked(store, ("pa", "member", member_id), None, limit=limit)
    return [
        {"key": it.key, "importance": (it.value or {}).get("importance"), "value": it.value}
        for it in items
    ]


def _compare_rows(multi_decision: Any, single_decision: Any) -> dict:
    """PR8 good-to-have: surfaces `pac compare`'s single-vs-multi-agent
    distinction in the UI, not just the CLI. Pure -- both decisions are
    already-validated PADecision objects or None."""
    multi = multi_decision.model_dump() if multi_decision is not None else None
    single = single_decision.model_dump() if single_decision is not None else None
    if multi is None or single is None:
        agreement = "incomplete -- at least one side has no decision yet"
    elif multi["disposition"] == single["disposition"]:
        agreement = f"same disposition ({multi['disposition']})"
    else:
        agreement = (
            f"different disposition: multi-agent={multi['disposition']!r} "
            f"single-agent={single['disposition']!r}"
        )
    return {"multi_agent": multi, "single_agent": single, "agreement": agreement}


async def _run_one_case(
    raw_provider_text: str, case_id: str, session_id: str, member_id: str
) -> tuple[dict, list[dict]]:
    from pa_copilot.cli import _open_graph, _open_state
    from pa_copilot.config import get_settings
    from pa_copilot.state import new_case_state

    settings = get_settings()
    async with _open_state(settings) as (checkpointer, store):
        async with _open_graph(store, checkpointer) as graph:
            state = new_case_state(case_id, session_id, member_id, raw_provider_text)
            thread = {"configurable": {"thread_id": case_id}}
            result = await graph.ainvoke(state, config=thread)
        return result, _long_term_hits(store, member_id)


async def _run_compare_case(
    raw_provider_text: str, case_id: str, session_id: str, member_id: str
) -> tuple[dict, Any, list[dict]]:
    """PR8 good-to-have: run both the multi-agent graph and the single-agent
    baseline over one case (same pairing `pac compare` already runs), sharing
    one MCP session/store the way cli.py's `compare()` does."""
    from pa_copilot.cli import _open_mcp_tools, _open_state
    from pa_copilot.config import get_settings
    from pa_copilot.graph import make_graph
    from pa_copilot.single_agent import run_single_agent
    from pa_copilot.state import new_case_state

    settings = get_settings()
    async with _open_state(settings) as (checkpointer, store):
        async with _open_mcp_tools() as mcp_tools:
            graph = await make_graph(store=store, checkpointer=checkpointer, mcp_tools=mcp_tools)
            state = new_case_state(case_id, session_id, member_id, raw_provider_text)
            thread = {"configurable": {"thread_id": f"{case_id}-multi"}}
            multi_result = await graph.ainvoke(state, config=thread)
            single_decision = await run_single_agent(
                raw_provider_text, member_id, mcp_tools=mcp_tools, store=store,
            )
        return multi_result, single_decision, _long_term_hits(store, member_id)


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
        compare_mode = st.checkbox(
            "Compare single-agent vs multi-agent",
            help="PR8: also runs the single-agent baseline (pac compare's pairing) "
            "over the same request and shows both decisions side by side.",
        )
        run_clicked = st.button("Run")
    with center:
        st.subheader("Routing trail")
        trail_placeholder = st.empty()
        st.subheader("Artifacts")
        artifact_placeholder = st.empty()
        st.subheader("Single- vs multi-agent comparison")
        compare_placeholder = st.empty()
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
        if compare_mode:
            result, single_decision, long_term_hits = asyncio.run(
                _run_compare_case(raw_provider_text, case_id, session_id, member_id)
            )
            compare_placeholder.json(_compare_rows(result.get("decision"), single_decision))
        else:
            result, long_term_hits = asyncio.run(
                _run_one_case(raw_provider_text, case_id, session_id, member_id)
            )
            compare_placeholder.caption("Enable \"Compare single-agent vs multi-agent\" to see this.")
    except Exception as exc:  # noqa: BLE001 -- surface in the UI, don't crash the app
        st.error(str(exc))
        return

    trail_placeholder.table(_route_rows(result.get("route_history") or []))
    artifact_placeholder.json(_artifact_rows(result))
    memory_placeholder.json(_memory_panel_rows(result.get("working_memory"), long_term_hits))


if __name__ == "__main__":
    main()
