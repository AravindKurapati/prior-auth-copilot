"""Task 5 (AC-12): shared scripted-case builders for the reflection/self-healing
loop's two triggers (tool failure, low confidence), each with a recovers/exhausts
variant. Not a test module itself (no ``test_*`` functions).

Reuses ``tests/_full_case.py``'s shared member/service/provider constants,
``new_initial_state``, and fake MCP tools rather than inventing new synthetic
IDs. Every case pre-seeds ``request`` (and, for the low-confidence cases,
``benefit`` too) directly into the initial state rather than scripting
``intake``/``benefit_check`` from scratch -- AC-02/AC-03 already prove those
stages thoroughly; this module's job is the reflection mechanism itself, kept
focused rather than re-deriving an entire case from raw text each time.

Tool-failure cases exercise ``benefit_check`` (bound to exactly one tool,
``benefit_lookup``, via a stateful flaky fake with a call counter) with
``request`` pre-seeded (``benefit=None`` so ``hard_route``'s existing
"benefit is None -> benefit_check" guardrail drives the whole reroute
deterministically -- no LLM router call needed anywhere in either variant).

Low-confidence cases exercise ``medical_necessity`` (``request``+``benefit``
pre-seeded so ``hard_route``'s "necessity is None -> medical_necessity" fires
first, then falls through to the LLM router for every subsequent turn once
``necessity`` is set) with a scripted ``RouterDecision`` explicitly choosing to
loop back to ``medical_necessity`` -- proving design.md's "supervisor loops
back" is a real, reachable path, not just a theoretical fall-through (see PR6
plan's Architecture section 3 and Task 3's report for why this is an LLM-router
choice, not a new hard_route rule).

Attempt counts (how many times a flaky/low-confidence tool or worker fires
before the scenario resolves) are read from ``get_settings()`` at call time
(``max_tool_retries``, ``max_replans``), not hardcoded, so a future config
change doesn't silently desync these scripts from the real cap.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from _fakes import FakeToolCallingModel, ai_tool_call
from _full_case import (
    EXPECTED_BENEFIT,
    EXPECTED_REQUEST,
    MEMBER_ID,
    SERVICE_CODE,
    fake_criteria_check,
    new_initial_state,
)
from pa_copilot.config import get_settings
from pa_copilot.schemas import NecessityAssessment, PADecision, RouterDecision

CASE_ID_TOOL_FAILURE_RECOVERS = "case-reflection-tool-failure-recovers"
CASE_ID_TOOL_FAILURE_EXHAUSTS = "case-reflection-tool-failure-exhausts"
CASE_ID_LOW_CONFIDENCE_RECOVERS = "case-reflection-low-confidence-recovers"
CASE_ID_LOW_CONFIDENCE_EXHAUSTS = "case-reflection-low-confidence-exhausts"


@dataclass
class ReflectionCase:
    case_id: str
    model: FakeToolCallingModel
    mcp_tools: list


def make_flaky_benefit_lookup(*, fail_times: int) -> tuple:
    """A benefit_lookup fake that raises for its first ``fail_times`` calls,
    then succeeds on every call after. ``fail_times`` huge (e.g. 10_000) never
    succeeds within any real test's lifetime -- the exhaustion variant's shape."""

    calls = {"n": 0}

    @tool("benefit_lookup")
    def flaky_benefit_lookup(member_id: str, service_code: str) -> dict:
        """Look up benefit coverage (flaky fake -- fails fail_times then succeeds)."""
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise RuntimeError(f"mcp transient failure (attempt {calls['n']})")
        return {
            "found": True, "covered": True, "requires_pa": True,
            "plan_id": "PLAN-GOLD-PPO", "network_status": "in_network",
        }

    return flaky_benefit_lookup, calls


def build_tool_failure_recovers_case() -> ReflectionCase:
    """benefit_check's tool fails exactly max_tool_retries times (all consumed
    within ONE node call's own tenacity attempts), then succeeds on the
    supervisor's deterministic reroute back to benefit_check (a SECOND node
    call). Proves the OUTER reflection loop (supervisor-level reroute after
    tool_failure_update), distinct from tenacity's inner retry (already
    covered by Task 1's test_reflection_resilience.py)."""
    s = get_settings()
    flaky_tool, _calls = make_flaky_benefit_lookup(fail_times=s.max_tool_retries)

    necessity = NecessityAssessment(
        criteria_status="met", policy_id="PA-MRI-LUMBAR", citations=[], unmet_requirements=[],
        confidence=0.9,
        rationale="8 weeks conservative therapy and persistent radicular pain documented; not excluded",
    )
    decision = PADecision(
        disposition="approve", cited_criteria=[],
        reviewer_summary="Criteria met on mechanical screen; recommend approval pending sign-off.",
        confidence=0.9, human_review_required=True,
    )
    script = (
        [ai_tool_call("benefit_lookup", {"member_id": MEMBER_ID, "service_code": SERVICE_CODE})]
        * s.max_tool_retries  # every attempt in the first (failing) node call
        + [
            ai_tool_call("benefit_lookup", {"member_id": MEMBER_ID, "service_code": SERVICE_CODE}),
            AIMessage(content="checked"),
            ai_tool_call("criteria_check", {"service_code": SERVICE_CODE, "diagnosis_codes": ["M54.16"]}),
            AIMessage(content="assessed"),
            AIMessage(content="drafted"),
        ]
    )
    model = FakeToolCallingModel(
        script=script,
        structured_responses=[
            EXPECTED_BENEFIT,
            necessity,
            RouterDecision(next="decision_draft", rationale="necessity clearly met; ready to draft"),
            decision,
        ],
    )
    return ReflectionCase(
        CASE_ID_TOOL_FAILURE_RECOVERS, model, [flaky_tool, fake_criteria_check]
    )


def build_tool_failure_exhausts_case() -> ReflectionCase:
    """benefit_lookup always fails. After max_replans reroutes (each burning
    max_tool_retries tenacity attempts internally), hard_route's broadened cap
    check fires deterministically -> human_review, with NO LLM router call
    needed anywhere in this scenario."""
    s = get_settings()
    flaky_tool, _calls = make_flaky_benefit_lookup(fail_times=10_000)
    total_attempts = s.max_tool_retries * s.max_replans
    script = [
        ai_tool_call("benefit_lookup", {"member_id": MEMBER_ID, "service_code": SERVICE_CODE})
    ] * total_attempts
    model = FakeToolCallingModel(script=script, structured_responses=[])
    return ReflectionCase(CASE_ID_TOOL_FAILURE_EXHAUSTS, model, [flaky_tool])


LOW_CONFIDENCE_NECESSITY = NecessityAssessment(
    criteria_status="indeterminate", policy_id="PA-MRI-LUMBAR", citations=[],
    unmet_requirements=["conservative-therapy duration not clearly documented"],
    confidence=0.2, rationale="mechanical check indeterminate; low confidence on first pass",
)
HIGH_CONFIDENCE_NECESSITY = NecessityAssessment(
    criteria_status="met", policy_id="PA-MRI-LUMBAR", citations=[], unmet_requirements=[],
    confidence=0.9, rationale="broader review confirms criteria met",
)


def build_low_confidence_recovers_case() -> ReflectionCase:
    """medical_necessity returns a low-confidence, indeterminate assessment on
    its first visit (needs_replan=True, per Task 3's medical_necessity.py);
    the supervisor's LLM router explicitly chooses to loop back to
    medical_necessity (design.md's "loops back with a hint" -- an LLM-router
    choice this scenario proves is reachable, not a hard_route rule); the
    second visit returns high confidence, and the case proceeds to
    decision_draft/FINISH normally."""
    low_necessity = LOW_CONFIDENCE_NECESSITY
    high_necessity = HIGH_CONFIDENCE_NECESSITY
    decision = PADecision(
        disposition="approve", cited_criteria=[],
        reviewer_summary="Criteria met after reassessment; recommend approval pending sign-off.",
        confidence=0.9, human_review_required=True,
    )
    model = FakeToolCallingModel(
        script=[
            ai_tool_call("criteria_check", {"service_code": SERVICE_CODE, "diagnosis_codes": ["M54.16"]}),
            AIMessage(content="assessed, low confidence"),
            ai_tool_call("criteria_check", {"service_code": SERVICE_CODE, "diagnosis_codes": ["M54.16"]}),
            AIMessage(content="reassessed"),
            AIMessage(content="drafted"),
        ],
        structured_responses=[
            low_necessity,
            RouterDecision(
                next="medical_necessity",
                rationale="low confidence on first pass; broadening retrieval and reassessing",
            ),
            high_necessity,
            RouterDecision(next="decision_draft", rationale="necessity now clearly met"),
            decision,
        ],
    )
    return ReflectionCase(CASE_ID_LOW_CONFIDENCE_RECOVERS, model, [fake_criteria_check])


def build_low_confidence_exhausts_case() -> ReflectionCase:
    """medical_necessity always returns low confidence. After max_replans
    loop-backs, hard_route's broadened cap check fires deterministically ->
    human_review, before the LLM router is ever consulted for that final turn
    (only max_replans - 1 RouterDecision loop-back choices are needed)."""
    s = get_settings()
    low_necessity = LOW_CONFIDENCE_NECESSITY
    script = []
    structured = []
    for i in range(s.max_replans):
        script += [
            ai_tool_call("criteria_check", {"service_code": SERVICE_CODE, "diagnosis_codes": ["M54.16"]}),
            AIMessage(content=f"assessed, low confidence (attempt {i + 1})"),
        ]
        structured.append(low_necessity)
        if i < s.max_replans - 1:
            structured.append(
                RouterDecision(
                    next="medical_necessity",
                    rationale="low confidence; broadening retrieval and reassessing",
                )
            )
    model = FakeToolCallingModel(script=script, structured_responses=structured)
    return ReflectionCase(CASE_ID_LOW_CONFIDENCE_EXHAUSTS, model, [fake_criteria_check])


def seed_request_only(case_id: str) -> dict:
    state = new_initial_state(case_id)
    state["request"] = EXPECTED_REQUEST
    return state


def seed_request_and_benefit(case_id: str) -> dict:
    state = seed_request_only(case_id)
    state["benefit"] = EXPECTED_BENEFIT
    return state
