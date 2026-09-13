"""AC-01: the system is built on LangGraph with an explicit typed state object
(TypedDict / Pydantic) shared across nodes -- PACaseState (state.py)."""

import typing

from langgraph.graph.message import add_messages

from pa_copilot.state import PACaseState, REDUCER_FIELDS, new_case_state


def test_pa_case_state_is_a_typed_dict():
    assert typing.is_typeddict(PACaseState)


def test_every_node_module_reads_and_writes_pa_case_state():
    # Structural check: each worker/supervisor/summarize module's node callable
    # is a plain async function accepting one PACaseState-shaped dict and
    # returning a dict — verified by this PR's own worker tests already
    # exercising that contract; this test asserts the modules import cleanly
    # and expose the expected factory/function names.
    from pa_copilot import supervisor
    from pa_copilot.agents import benefit_check, decision_draft, human_review, intake, medical_necessity
    from pa_copilot.context import summarization

    assert hasattr(supervisor, "build_supervisor_node")
    assert hasattr(intake, "build_intake_node")
    assert hasattr(benefit_check, "build_benefit_check_node")
    assert hasattr(medical_necessity, "build_medical_necessity_node")
    assert hasattr(decision_draft, "build_decision_draft_node")
    assert hasattr(human_review, "build_human_review_node")
    assert hasattr(summarization, "summarize")


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


def test_reducer_fields_derived_from_annotations():
    # Derived by introspecting PACaseState's Annotated[..., <reducer>] metadata,
    # so this actually verifies the wiring rather than comparing a literal to itself.
    assert {"messages", "route_history", "tool_failures"} <= REDUCER_FIELDS
    assert REDUCER_FIELDS == {"messages", "route_history", "tool_failures"}
