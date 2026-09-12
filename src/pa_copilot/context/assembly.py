"""Write/Select context-engineering strategies (design.md §5). select_for is the
"Select" row: the minimal per-node field view, computed once here rather than
re-derived ad hoc inside each worker. write_working_memory is the "Write" row's
scratch-memory helper, a thin state-update wrapper over memory.working.remember."""

from __future__ import annotations

from typing import Any

from pa_copilot.memory import working
from pa_copilot.state import PACaseState

_SELECTORS: dict[str, tuple[str, ...]] = {
    "intake": ("raw_provider_text", "quarantine_ref", "member_id", "case_id"),
    "benefit_check": ("request",),
    "medical_necessity": ("request", "benefit", "retrieved_criteria"),
    "decision_draft": ("request", "benefit", "necessity"),
}


def select_for(node: str, state: PACaseState) -> dict[str, Any]:
    fields = _SELECTORS.get(node)
    if fields is None:
        raise ValueError(f"no field selection defined for node {node!r}")
    result = {f: state.get(f) for f in fields}
    if node == "medical_necessity":
        result["retrieved_criteria"] = result.get("retrieved_criteria") or []
    return result


def write_working_memory(state: PACaseState, key: str, value: Any) -> dict[str, Any]:
    return {"working_memory": working.remember(state.get("working_memory"), key, value)}
