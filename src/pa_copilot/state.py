from __future__ import annotations

import operator
import typing
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from pa_copilot.schemas import (
    BenefitResult,
    CriteriaCitation,
    NecessityAssessment,
    PADecision,
    PARequest,
    RouteStep,
    ToolFailure,
)

_REDUCERS = (operator.add, add_messages)


class PACaseState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    case_id: str
    session_id: str
    member_id: str
    raw_provider_text: str
    quarantine_ref: str
    request: PARequest
    benefit: BenefitResult
    necessity: NecessityAssessment
    decision: PADecision
    retrieved_criteria: list[CriteriaCitation]
    next: str
    route_history: Annotated[list[RouteStep], operator.add]
    confidence: float
    needs_replan: bool
    replan_count: int
    supervisor_hops: int
    tool_failures: Annotated[list[ToolFailure], operator.add]
    working_memory: dict[str, Any]
    context: dict[str, Any]


def _derive_reducer_fields() -> set[str]:
    """The append-reduced fields, read straight off `PACaseState`'s
    `Annotated[..., <reducer>]` metadata rather than a hand-maintained literal."""
    hints = typing.get_type_hints(PACaseState, include_extras=True)
    fields: set[str] = set()
    for name, hint in hints.items():
        if any(meta in _REDUCERS for meta in typing.get_args(hint)[1:]):
            fields.add(name)
    return fields


REDUCER_FIELDS: set[str] = _derive_reducer_fields()


def new_case_state(
    case_id: str, session_id: str, member_id: str, raw_provider_text: str
) -> PACaseState:
    return PACaseState(
        messages=[],
        case_id=case_id,
        session_id=session_id,
        member_id=member_id,
        raw_provider_text=raw_provider_text,
        retrieved_criteria=[],
        route_history=[],
        needs_replan=False,
        replan_count=0,
        supervisor_hops=0,
        tool_failures=[],
        working_memory={},
        context={},
    )
