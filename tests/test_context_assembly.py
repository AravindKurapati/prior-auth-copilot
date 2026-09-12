import pytest

from pa_copilot.context.assembly import select_for, write_working_memory


def _state(**kw):
    return {
        "raw_provider_text": "raw text",
        "quarantine_ref": "quarantine:c1",
        "member_id": "M1",
        "case_id": "c1",
        "request": {"service_code": "72148"},
        "benefit": {"covered": True},
        "necessity": {"criteria_status": "met"},
        "retrieved_criteria": [{"clause_id": "x"}],
        "working_memory": {},
        **kw,
    }


def test_select_for_intake_excludes_downstream_fields():
    got = select_for("intake", _state())
    assert set(got) == {"raw_provider_text", "quarantine_ref", "member_id", "case_id"}


def test_select_for_benefit_check_is_request_only():
    got = select_for("benefit_check", _state())
    assert got == {"request": {"service_code": "72148"}}


def test_select_for_decision_draft_never_includes_raw_text():
    got = select_for("decision_draft", _state())
    assert "raw_provider_text" not in got
    assert set(got) == {"request", "benefit", "necessity"}


def test_select_for_medical_necessity_defaults_retrieved_criteria():
    got = select_for("medical_necessity", _state(retrieved_criteria=None))
    assert got["retrieved_criteria"] == []


def test_select_for_unknown_node_raises():
    with pytest.raises(ValueError, match="no field selection"):
        select_for("nonexistent_node", _state())


def test_write_working_memory_is_a_state_update_not_a_mutation():
    st = _state()
    update = write_working_memory(st, "seen_service_code", "72148")
    assert st["working_memory"] == {}  # original untouched
    assert update == {"working_memory": {"facts": {"seen_service_code": "72148"}}}
