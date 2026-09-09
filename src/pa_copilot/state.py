from __future__ import annotations

import operator
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

REDUCER_FIELDS: set[str] = {"messages", "route_history", "tool_failures"}


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
