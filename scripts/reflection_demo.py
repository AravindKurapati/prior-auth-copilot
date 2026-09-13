"""AC-12 reflection/self-healing evidence (Task 5). Runs the two RECOVER
scenarios through the REAL compiled graph (pa_copilot.graph.make_graph) --
fakes throughout, deterministic, no API key needed:

    (a) tool-failure recovers -> traces/reflection_tool_failure.json
    (b) low-confidence recovers -> traces/reflection_low_confidence.json

The EXHAUSTS variants of both scenarios are proven in
tests/test_ac12_reflection.py but not separately committed as trace files --
one illustrative recovery trace per trigger is the evidence artifact; the
exhaustion path's behavior (never an unhandled exception, clean route to
human_review) is what NFR-07's own test asserts, not a distinct trace shape.

Reuses scripts/full_case_demo.py's exact byte-stable-serialization helpers
(_dump_route_step strips the non-deterministic `ts` field; _write pins
utf-8/\\n) rather than re-deriving them -- see that module for the reasoning
(PR5b's own final-review finding about timestamps leaking into committed
traces).

    python scripts/reflection_demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "tests"))


def _dump_model(obj) -> dict | None:
    return obj.model_dump() if obj is not None else None


async def _run_tool_failure_recovers() -> dict:
    from langgraph.checkpoint.memory import InMemorySaver  # noqa: PLC0415

    from _reflection_case import (  # noqa: PLC0415
        build_tool_failure_recovers_case,
        seed_request_only,
    )
    from full_case_demo import _open_memory_store  # noqa: PLC0415
    from pa_copilot.context import summarization as summarization_mod  # noqa: PLC0415
    from pa_copilot.graph import make_graph  # noqa: PLC0415

    from _full_case import build_stub_summarization_node  # noqa: PLC0415

    summarization_mod._node = build_stub_summarization_node()

    store = _open_memory_store()
    try:
        case = build_tool_failure_recovers_case()
        graph = await make_graph(
            store=store, checkpointer=InMemorySaver(), mcp_tools=case.mcp_tools, model=case.model
        )
        thread = {"configurable": {"thread_id": case.case_id}}
        result = await graph.ainvoke(seed_request_only(case.case_id), config=thread)
        return result
    finally:
        store.conn.close()


async def _run_low_confidence_recovers() -> dict:
    from langgraph.checkpoint.memory import InMemorySaver  # noqa: PLC0415

    from _reflection_case import (  # noqa: PLC0415
        build_low_confidence_recovers_case,
        seed_request_and_benefit,
    )
    from full_case_demo import _open_memory_store  # noqa: PLC0415
    from pa_copilot.context import summarization as summarization_mod  # noqa: PLC0415
    from pa_copilot.graph import make_graph  # noqa: PLC0415

    from _full_case import build_stub_summarization_node  # noqa: PLC0415

    summarization_mod._node = build_stub_summarization_node()

    store = _open_memory_store()
    try:
        case = build_low_confidence_recovers_case()
        graph = await make_graph(
            store=store, checkpointer=InMemorySaver(), mcp_tools=case.mcp_tools, model=case.model
        )
        thread = {"configurable": {"thread_id": case.case_id}}
        result = await graph.ainvoke(seed_request_and_benefit(case.case_id), config=thread)
        return result
    finally:
        store.conn.close()


def main() -> None:
    from full_case_demo import TRACE_SCHEMA_VERSION, _dump_route_step, _write  # noqa: PLC0415

    tool_failure_result = asyncio.run(_run_tool_failure_recovers())
    assert "__interrupt__" not in tool_failure_result, "tool-failure case unexpectedly paused"
    assert tool_failure_result["next"] == "FINISH"
    tool_failure_doc = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "case_id": tool_failure_result["case_id"],
        "route_history": [_dump_route_step(step) for step in tool_failure_result["route_history"]],
        "tool_failures": [
            {k: v for k, v in f.model_dump().items() if k != "ts"}
            for f in tool_failure_result["tool_failures"]
        ],
        "replan_count": tool_failure_result["replan_count"],
        "decision": _dump_model(tool_failure_result.get("decision")),
    }
    path_tool_failure = _write("reflection_tool_failure.json", tool_failure_doc)
    print(f"wrote {path_tool_failure.relative_to(_REPO_ROOT).as_posix()}")
    print(f"  tool_failures: {len(tool_failure_result['tool_failures'])}, replan_count: {tool_failure_result['replan_count']}")

    low_confidence_result = asyncio.run(_run_low_confidence_recovers())
    assert "__interrupt__" not in low_confidence_result, "low-confidence case unexpectedly paused"
    assert low_confidence_result["next"] == "FINISH"
    from _reflection_case import HIGH_CONFIDENCE_NECESSITY, LOW_CONFIDENCE_NECESSITY  # noqa: PLC0415

    low_confidence_doc = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "case_id": low_confidence_result["case_id"],
        "route_history": [_dump_route_step(step) for step in low_confidence_result["route_history"]],
        "first_assessment": _dump_model(LOW_CONFIDENCE_NECESSITY),
        "final_assessment": _dump_model(low_confidence_result.get("necessity")),
        "replan_count": low_confidence_result["replan_count"],
        "decision": _dump_model(low_confidence_result.get("decision")),
    }
    assert low_confidence_doc["final_assessment"] == _dump_model(HIGH_CONFIDENCE_NECESSITY)
    path_low_confidence = _write("reflection_low_confidence.json", low_confidence_doc)
    print(f"wrote {path_low_confidence.relative_to(_REPO_ROOT).as_posix()}")
    print(f"  replan_count: {low_confidence_result['replan_count']}")


if __name__ == "__main__":
    main()
