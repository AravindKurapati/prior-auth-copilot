"""PACaseState field contract. summarized_messages must NOT be a reducer field —
context/summarization.py (Task 4) replaces the whole compressed view each call,
it does not append to it."""

from pa_copilot.state import REDUCER_FIELDS, new_case_state


def test_new_case_state_seeds_summarized_messages_empty():
    st = new_case_state("c1", "s1", "M1", "raw text")
    assert st["summarized_messages"] == []


def test_summarized_messages_is_not_a_reducer_field():
    assert "summarized_messages" not in REDUCER_FIELDS
    assert "route_history" in REDUCER_FIELDS  # sanity check the derivation still works
    assert "messages" in REDUCER_FIELDS
