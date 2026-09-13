"""Pure-function surface of app/streamlit_app.py -- no Streamlit runtime needed
(AppTest is out of scope for this PR, see docs/implementation-plan-pr7.md's
Design decisions)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from streamlit_app import _artifact_rows, _memory_panel_rows, _route_rows  # noqa: E402

from pa_copilot.schemas import BenefitResult, RouteStep  # noqa: E402


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
