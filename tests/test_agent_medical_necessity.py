"""medical_necessity worker: calls MCP criteria_check, decides agentically
whether to call the agentic RAG tool search_clinical_guidance (the system
prompt teaches design.md §4.1's status mapping — not a hard code gate), emits
a validated NecessityAssessment."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from _fakes import FakeToolCallingModel, ai_tool_call
from pa_copilot.agents.medical_necessity import build_medical_necessity_node
from pa_copilot.schemas import BenefitResult, CriteriaCitation, NecessityAssessment, PARequest


def _state(**kw) -> dict:
    request = PARequest(
        member_id="M100001", service_code="72148", diagnosis_codes=["M54.16"],
        requested_units=1, place_of_service="outpatient", provider_npi="1093817465",
        clinical_summary="MRI lumbar spine", missing_fields=[],
    )
    benefit = BenefitResult(covered=True, plan_id="PPO-100", requires_pa=True, network_status="in_network")
    return {"request": request, "benefit": benefit, "retrieved_criteria": [], **kw}


@tool
def criteria_check(service_code: str, diagnosis_codes: list) -> dict:
    """Check criteria."""
    return {"found": True, "policy_id": "PA-MRI-LUMBAR", "status": "indeterminate",
            "required_conditions": ["8 weeks conservative therapy"], "exclusions": []}


@pytest.mark.asyncio
async def test_medical_necessity_calls_rag_when_indeterminate():
    citation = CriteriaCitation(source="rag_corpus", clause_id="c1", quote="8 weeks PT required", relevance="high")
    expected = NecessityAssessment(criteria_status="met", policy_id="PA-MRI-LUMBAR",
                                    citations=[citation], unmet_requirements=[], confidence=0.8,
                                    rationale="PT documented")
    model = FakeToolCallingModel(
        script=[
            ai_tool_call("criteria_check", {"service_code": "72148", "diagnosis_codes": ["M54.16"]}),
            ai_tool_call("search_clinical_guidance", {"query": "8 weeks PT", "service_code": "72148"}),
            AIMessage(content="assessed"),
        ],
        structured_responses=[expected],
    )
    node = build_medical_necessity_node(mcp_tools=[criteria_check], model=model)
    update = await node(_state())
    assert update["necessity"] == expected
    assert update["retrieved_criteria"] == [citation]


@pytest.mark.asyncio
async def test_medical_necessity_tool_failure_sets_needs_replan():
    # Named "criteria_check" explicitly (matching the scripted ai_tool_call
    # name) rather than left as the plain function-derived
    # "broken_criteria_check" — a mismatch here would make ToolNode treat the
    # call as an unknown-tool dispatch instead of actually invoking this
    # function, which would make the test pass for the wrong reason (script
    # exhaustion, not the tool's RuntimeError). Same naming detail flagged in
    # task-8-brief.md / test_agent_benefit_check.py.
    @tool("criteria_check")
    def broken_criteria_check(service_code: str, diagnosis_codes: list) -> dict:
        """Broken."""
        raise RuntimeError("mcp timeout")

    model = FakeToolCallingModel(
        script=[ai_tool_call("criteria_check", {"service_code": "72148", "diagnosis_codes": ["M54.16"]})],
        structured_responses=[],
    )
    node = build_medical_necessity_node(mcp_tools=[broken_criteria_check], model=model)
    update = await node(_state())
    assert update["needs_replan"] is True


@pytest.mark.asyncio
async def test_medical_necessity_defensive_fallback_when_benefit_missing():
    """PR5b final-review Fix B, part 2: hard_route's new benefit-None ->
    benefit_check guardrail is what actually prevents this in practice, but
    this node must not itself crash dereferencing a None benefit if it is
    ever reached out of order — belt-and-suspenders, not a substitute for the
    routing fix. model=None here: if the node tried to proceed to
    run_worker_react instead of returning immediately, it would blow up
    calling get_agent_model() with no GEMINI_API_KEY configured, so reaching
    the assertion at all proves the early return fired."""
    node = build_medical_necessity_node(mcp_tools=[criteria_check], model=None)
    update = await node(_state(benefit=None))
    assert update == {"needs_replan": True}
