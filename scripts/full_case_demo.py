"""AC-02/03/10/11 full in-graph evidence (Task 7).

Runs two full cases through the REAL compiled graph (`pa_copilot.graph.make_graph`)
-- fakes throughout (a scripted `FakeToolCallingModel` for the LLM, fake MCP
tools), deterministic, no API key needed:

    (a) a clear-cut case that reaches `decision_draft`/`FINISH` via >=2 workers
        -> `traces/run_full_case.json` (full final state: route_history +
        request/benefit/necessity/decision) and `traces/route_clearcut.json`
        (just the routing trail, extracted from the SAME run);
    (b) an ambiguous/low-confidence case that routes to `human_review` instead
        of an auto-draft -> `traces/route_ambiguous.json`.

Trace format decision: the brief flagged this as an open question -- a true
`RunTracer`-shaped trace needs a tracer callback threaded through every worker
node, which this task is explicitly barred from adding to `graph.py`/the worker
files. So this script takes the brief's offered simpler alternative: a direct
dump of `route_history` (each step's `.model_dump()`) plus the final
`request`/`benefit`/`necessity`/`decision` (each `.model_dump()` if present),
under the SAME `schema_version` key `RunTracer`/`tracing.py` uses, so a future
schema-unifying pass (PR7's `test_nfr04_trace_schema.py`) has a stable key to
migrate from rather than an unrelated one-off shape.

    python scripts/full_case_demo.py

``sys.path[0]`` is ``scripts/`` when invoked this way, so put the repo root
(and ``src``) on the path first, then reach into ``tests/`` for the shared
fakes and full-case scripting the same way ``scripts/run_pause_resume_test.py``
and ``scripts/graph_topology_demo.py`` already do.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "tests"))

TRACE_SCHEMA_VERSION = "1"


def _dump_model(obj) -> dict | None:
    return obj.model_dump() if obj is not None else None


def _dump_route_step(step) -> dict:
    """`RouteStep.model_dump()` minus `ts` -- `supervisor.py` stamps `ts` with
    `datetime.now(timezone.utc).isoformat()`, real wall-clock time, which would
    make every committed trace file under `traces/` non-byte-stable across
    regenerations (this project's committed-evidence convention, used by every
    other file in `traces/`, is "re-run the script, `git status` shows
    nothing changed"). Everything else in a `RouteStep` (`from_node`, `to_node`,
    `reason`) is fully deterministic given a scripted case."""
    return {k: v for k, v in step.model_dump().items() if k != "ts"}


async def _run_clear_cut() -> dict:
    from langgraph.checkpoint.memory import InMemorySaver  # noqa: PLC0415

    from _full_case import (  # noqa: PLC0415
        FAKE_MCP_TOOLS,
        build_clear_cut_case,
        build_stub_summarization_node,
        new_initial_state,
    )
    from pa_copilot.context import summarization as summarization_mod  # noqa: PLC0415
    from pa_copilot.graph import make_graph  # noqa: PLC0415

    # Standalone script (not pytest) -- set the module-level cache directly,
    # same effect as `_full_case.stub_summarizer(monkeypatch)` but with nothing
    # to restore since this process exits right after.
    summarization_mod._node = build_stub_summarization_node()

    store = _open_memory_store()
    try:
        case = build_clear_cut_case()
        graph = await make_graph(
            store=store, checkpointer=InMemorySaver(), mcp_tools=FAKE_MCP_TOOLS, model=case.model
        )
        thread = {"configurable": {"thread_id": case.case_id}}
        result = await graph.ainvoke(new_initial_state(case.case_id), config=thread)
        return result
    finally:
        store.conn.close()


def _open_memory_store():
    """The `intake` worker binds a memory-search tool, which needs a real
    `PolicyStore` -- an in-memory sqlite one is enough here (no persistence
    needed across this one-shot script run). Returns an OPEN store; the caller
    is responsible for closing `store.conn` (mirrors
    `pa_copilot.memory.store.memory_store`'s own contextmanager body, just
    without the `with`, since a plain function reads better at the two call
    sites below than repeating the `with` block twice)."""
    from _fakes import FakeEmbedder  # noqa: PLC0415
    from pa_copilot.memory.store import open_memory_store  # noqa: PLC0415

    return open_memory_store(":memory:", embedder=FakeEmbedder())


async def _run_ambiguous() -> dict:
    from langgraph.checkpoint.memory import InMemorySaver  # noqa: PLC0415

    from _full_case import (  # noqa: PLC0415
        FAKE_MCP_TOOLS,
        build_ambiguous_case,
        build_stub_summarization_node,
        fake_rag_embedder,
        new_initial_state,
    )
    from pa_copilot.context import summarization as summarization_mod  # noqa: PLC0415
    from pa_copilot.graph import make_graph  # noqa: PLC0415

    summarization_mod._node = build_stub_summarization_node()

    store = _open_memory_store()
    try:
        case = build_ambiguous_case()
        graph = await make_graph(
            store=store, checkpointer=InMemorySaver(), mcp_tools=FAKE_MCP_TOOLS, model=case.model
        )
        thread = {"configurable": {"thread_id": case.case_id}}
        with fake_rag_embedder():
            result = await graph.ainvoke(new_initial_state(case.case_id), config=thread)
        return result
    finally:
        store.conn.close()


def _route_history_doc(case_id: str, result: dict) -> dict:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "case_id": case_id,
        "route_history": [_dump_route_step(step) for step in result.get("route_history") or []],
        "final_next": result.get("next"),
        "paused_for_human_review": "__interrupt__" in result,
    }


def _write(name: str, doc: dict) -> Path:
    out = _REPO_ROOT / "traces" / name
    out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    return out


async def main() -> None:
    clearcut_result = await _run_clear_cut()
    assert "__interrupt__" not in clearcut_result, "clear-cut case unexpectedly paused"
    assert clearcut_result["next"] == "FINISH"
    visited = {s.to_node for s in clearcut_result["route_history"] if s.to_node != "FINISH"}
    assert len(visited) >= 2, f"clear-cut case only visited {visited}, expected >=2 workers"

    run_full_case_doc = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "case_id": clearcut_result["case_id"],
        "route_history": [_dump_route_step(step) for step in clearcut_result["route_history"]],
        "request": _dump_model(clearcut_result.get("request")),
        "benefit": _dump_model(clearcut_result.get("benefit")),
        "necessity": _dump_model(clearcut_result.get("necessity")),
        "decision": _dump_model(clearcut_result.get("decision")),
    }
    path_full = _write("run_full_case.json", run_full_case_doc)
    path_clearcut = _write(
        "route_clearcut.json", _route_history_doc(clearcut_result["case_id"], clearcut_result)
    )
    print(f"wrote {path_full.relative_to(_REPO_ROOT).as_posix()}")
    print(f"wrote {path_clearcut.relative_to(_REPO_ROOT).as_posix()}")
    print(f"  clear-cut route_history workers visited: {sorted(visited)}")
    print(f"  clear-cut decision: {clearcut_result['decision'].disposition}")

    ambiguous_result = await _run_ambiguous()
    assert "__interrupt__" in ambiguous_result, "ambiguous case unexpectedly did not pause"
    assert ambiguous_result.get("decision") is None
    path_ambiguous = _write(
        "route_ambiguous.json", _route_history_doc(ambiguous_result["case_id"], ambiguous_result)
    )
    print(f"wrote {path_ambiguous.relative_to(_REPO_ROOT).as_posix()}")
    print(
        "  ambiguous route_history workers visited: "
        f"{sorted({s.to_node for s in ambiguous_result['route_history']})}"
    )


if __name__ == "__main__":
    asyncio.run(main())
