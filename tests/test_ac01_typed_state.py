import typing

from langgraph.graph.message import add_messages

from pa_copilot.state import PACaseState, REDUCER_FIELDS, new_case_state


def test_state_is_typeddict_with_expected_keys():
    hints = typing.get_type_hints(PACaseState, include_extras=True)
    for key in [
        "messages", "case_id", "session_id", "member_id", "raw_provider_text",
        "quarantine_ref", "request", "benefit", "necessity", "decision",
        "retrieved_criteria", "next", "route_history", "confidence", "needs_replan",
        "replan_count", "supervisor_hops", "tool_failures", "working_memory", "context",
    ]:
        assert key in hints, f"missing state key: {key}"


def test_messages_uses_add_messages_reducer():
    hints = typing.get_type_hints(PACaseState, include_extras=True)
    meta = typing.get_args(hints["messages"])[1:]
    assert add_messages in meta


def test_factory_seeds_collections():
    s = new_case_state("c1", "s1", "M1", "raw text")
    assert s["case_id"] == "c1"
    assert s["messages"] == []
    assert s["route_history"] == []
    assert s["replan_count"] == 0
    assert s["supervisor_hops"] == 0
    assert s["needs_replan"] is False
    assert s["working_memory"] == {}


def test_reducer_fields_declared():
    assert REDUCER_FIELDS == {"messages", "route_history", "tool_failures"}
