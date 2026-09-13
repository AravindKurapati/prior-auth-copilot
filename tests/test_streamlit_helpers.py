"""Pure-function surface of app/streamlit_app.py -- no Streamlit runtime needed
(AppTest is out of scope for this PR, see docs/implementation-plan-pr7.md's
Design decisions)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from streamlit_app import (  # noqa: E402
    _artifact_rows,
    _compare_rows,
    _long_term_hits,
    _memory_panel_rows,
    _route_rows,
)

from pa_copilot.schemas import BenefitResult, PADecision, RouteStep  # noqa: E402


def test_route_rows_flattens_route_history():
    steps = [RouteStep(from_node="supervisor", to_node="intake", reason="first", ts="t1")]
    rows = _route_rows(steps)
    assert rows == [{"from": "supervisor", "to": "intake", "reason": "first"}]


def test_artifact_rows_dumps_present_fields_and_nones_missing():
    state = {
        "request": None,
        "benefit": BenefitResult(
            covered=True, plan_id="p", requires_pa=True, network_status="in_network"
        ),
    }
    rows = _artifact_rows(state)
    assert rows["request"] is None
    assert rows["benefit"]["plan_id"] == "p"
    assert rows["necessity"] is None
    assert rows["decision"] is None


def test_memory_panel_rows_defaults_to_empty():
    assert _memory_panel_rows(None, None) == {"working_memory": {}, "long_term_hits": []}


def test_memory_panel_rows_passes_through_given_values():
    wm = {"k": "v"}
    hits = [{"key": "det-1"}]
    assert _memory_panel_rows(wm, hits) == {"working_memory": wm, "long_term_hits": hits}


def test_long_term_hits_reads_ranked_member_memories(memory_store):
    """PR8: the memory panel used to always render an empty long_term_hits list."""
    memory_store.put(
        ("pa", "member", "M100001"), "det-1",
        {"content": "prior denial", "importance": "critical"},
    )
    hits = _long_term_hits(memory_store, "M100001")
    assert len(hits) == 1
    assert hits[0]["key"] == "det-1"
    assert hits[0]["importance"] == "critical"


def test_long_term_hits_empty_for_unknown_member(memory_store):
    assert _long_term_hits(memory_store, "unknown") == []


def _decision(disposition: str) -> PADecision:
    return PADecision(disposition=disposition, cited_criteria=[], reviewer_summary="s", confidence=0.9)


def test_compare_rows_flags_same_disposition():
    out = _compare_rows(_decision("approve"), _decision("approve"))
    assert out["agreement"] == "same disposition (approve)"


def test_compare_rows_flags_different_disposition():
    out = _compare_rows(_decision("approve"), _decision("deny"))
    assert "different disposition" in out["agreement"]
    assert out["multi_agent"]["disposition"] == "approve"
    assert out["single_agent"]["disposition"] == "deny"


def test_compare_rows_flags_incomplete_when_either_side_missing():
    assert _compare_rows(None, _decision("approve"))["agreement"].startswith("incomplete")
    assert _compare_rows(_decision("approve"), None)["agreement"].startswith("incomplete")
