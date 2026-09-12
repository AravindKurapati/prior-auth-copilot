import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from _fakes import FakeToolCallingModel, ai_tool_call
from pa_copilot.agents.benefit_check import build_benefit_check_node
from pa_copilot.schemas import BenefitResult, PARequest


def _state() -> dict:
    request = PARequest(
        member_id="M100001", service_code="72148", diagnosis_codes=["M54.16"],
        requested_units=1, place_of_service="outpatient", provider_npi="1093817465",
        clinical_summary="MRI lumbar spine", missing_fields=[],
    )
    return {"request": request, "working_memory": {}}


@tool
def benefit_lookup(member_id: str, service_code: str) -> dict:
    """Look up benefit coverage."""
    return {"found": True, "covered": True, "requires_pa": True, "plan_id": "PPO-100", "network_status": "in_network"}


@pytest.mark.asyncio
async def test_benefit_check_happy_path():
    expected = BenefitResult(covered=True, plan_id="PPO-100", requires_pa=True, network_status="in_network")
    model = FakeToolCallingModel(
        script=[ai_tool_call("benefit_lookup", {"member_id": "M100001", "service_code": "72148"}),
                AIMessage(content="checked")],
        structured_responses=[expected],
    )
    node = build_benefit_check_node(mcp_tools=[benefit_lookup], model=model)
    update = await node(_state())
    assert update["benefit"] == expected


@pytest.mark.asyncio
async def test_benefit_check_tool_failure_sets_needs_replan():
    # Named "benefit_lookup" explicitly (matching the scripted ai_tool_call name)
    # rather than left as the plain function-derived name — a mismatch here
    # would make ToolNode treat the call as an unknown-tool dispatch instead of
    # actually invoking this function, which would make the test pass for the
    # wrong reason (script exhaustion, not the tool's RuntimeError). Same
    # naming detail flagged in task-8-brief.md for the fake provider_lookup.
    @tool("benefit_lookup")
    def broken_benefit_lookup(member_id: str, service_code: str) -> dict:
        """Broken benefit lookup."""
        raise RuntimeError("timeout")

    model = FakeToolCallingModel(
        script=[ai_tool_call("benefit_lookup", {"member_id": "M100001", "service_code": "72148"})],
        structured_responses=[],
    )
    node = build_benefit_check_node(mcp_tools=[broken_benefit_lookup], model=model)
    update = await node(_state())
    assert update["needs_replan"] is True
