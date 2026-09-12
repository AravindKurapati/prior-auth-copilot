import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import ValidationError

from _fakes import FakeToolCallingModel, ai_tool_call
from pa_copilot.agents._react import WorkerOutputError
from pa_copilot.agents.decision_draft import build_decision_draft_node
from pa_copilot.agents.intake import build_intake_node
from pa_copilot.schemas import (
    BenefitResult,
    CriteriaCitation,
    NecessityAssessment,
    PADecision,
    PARequest,
    RouteStep,
    RouterDecision,
    SampleSubmission,
    ToolFailure,
)

_SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_parequest_roundtrips():
    r = PARequest(
        member_id="M1",
        service_code="72148",
        diagnosis_codes=["M54.16"],
        requested_units=1,
        place_of_service="outpatient",
        provider_npi="1093817465",
        clinical_summary="8wks PT, radicular pain",
    )
    assert r.missing_fields == []
    assert PARequest.model_validate(r.model_dump()) == r


def test_router_decision_rejects_unknown_target():
    with pytest.raises(ValidationError):
        RouterDecision(next="frobnicate", rationale="nope")


def test_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        NecessityAssessment(
            criteria_status="met", policy_id="PA-MRI-001",
            confidence=1.4, rationale="x",
        )


def test_padecision_defaults_to_human_review():
    d = PADecision(disposition="deny", reviewer_summary="criteria not met", confidence=0.8)
    assert d.human_review_required is True


def test_citation_source_constrained():
    with pytest.raises(ValidationError):
        CriteriaCitation(source="wikipedia", clause_id="c1", quote="q", relevance="r")


def test_sample_submission_validates_every_committed_sample():
    files = sorted(_SAMPLES_DIR.glob("*.json"))
    assert files, "no committed samples found"
    for f in files:
        sub = SampleSubmission.model_validate(json.loads(f.read_text(encoding="utf-8")))
        assert sub.case_id and sub.session_id and sub.member_id
        assert isinstance(sub.structured, dict)


def test_sample_submission_rejects_missing_fields():
    with pytest.raises(ValidationError):
        SampleSubmission(case_id="c", session_id="s", member_id="M1")


def test_provenance_models():
    assert ToolFailure(tool="criteria_check", error="timeout", attempt=1, ts="t").attempt == 1
    assert RouteStep(from_node="supervisor", to_node="intake", reason="no request", ts="t")
    assert BenefitResult(covered=True, plan_id="P1", requires_pa=True, network_status="in")


# --- Task 8: AC-04 full closure ---------------------------------------------
# Everything above this line (PR1, Task 3) proves AC-04's "validated structured
# objects" half at the schema-unit level only. The acceptance-criteria row's
# test contract also requires "malformed output raises ValidationError and is
# handled" -- these two tests prove that at the WORKER level (not just the
# shared _react.py helper -- tests/test_react_helper.py already covers that in
# PR5a): a real pydantic.ValidationError from a malformed structured_responses
# entry surfaces through a real build_*_node(...) call as WorkerOutputError,
# never silently swallowed and never mislabeled as a WorkerToolError.
#
# The base FakeToolCallingModel.with_structured_output(...) (tests/_fakes.py)
# just returns whatever is in structured_responses verbatim -- correct for
# every other test in this suite, which always scripts an already-valid model
# instance, but it would never actually exercise pydantic validation for a
# malformed entry. ValidatingFakeModel below re-validates the popped entry
# against the schema, exactly like a real model.with_structured_output(schema)
# call does against a live model's output, so the ValidationError raised here
# is genuine (triggered by a missing required field), not hand-raised by the
# test.
class ValidatingFakeModel(FakeToolCallingModel):
    def with_structured_output(self, schema, **kwargs):
        def _pop_and_validate(*_args, **_kwargs):
            assert self.structured_responses, "FakeToolCallingModel structured_responses exhausted"
            raw = self.structured_responses.pop(0)
            return raw if isinstance(raw, schema) else schema.model_validate(raw)

        return RunnableLambda(_pop_and_validate)


@tool("provider_lookup")
def _fake_provider_lookup(npi: str) -> dict:
    """Look up a provider (fake MCP tool)."""
    return {"found": True, "npi": npi, "name": "Dr. Pat Vega", "network_status": "in_network"}


@pytest.mark.asyncio
async def test_intake_worker_output_error_propagates_not_swallowed(memory_store):
    """intake.py only catches WorkerToolError (src/pa_copilot/agents/intake.py)
    -- WorkerOutputError is not caught there, so a malformed structured-output
    entry must propagate out of build_intake_node's node function as a real
    WorkerOutputError. If it were silently swallowed, or mislabeled as a
    WorkerToolError, this would either raise nothing (or the wrong exception
    type) instead."""
    model = ValidatingFakeModel(
        script=[ai_tool_call("provider_lookup", {"npi": "1093817465"}), AIMessage(content="extracted")],
        # Missing every required PARequest field except one -- a genuine
        # pydantic.ValidationError, not a hand-raised one.
        structured_responses=[{"clinical_summary": "MRI lumbar spine"}],
    )
    node = build_intake_node(store=memory_store, mcp_tools=[_fake_provider_lookup], model=model)
    state = {
        "case_id": "case-ac04-1", "session_id": "sess-ac04-1", "member_id": "M100001",
        "raw_provider_text": "MRI lumbar spine, dx M54.16, NPI 1093817465.", "working_memory": {},
    }
    with pytest.raises(WorkerOutputError) as exc_info:
        await node(state)
    assert "validation error" in exc_info.value.error.lower()


@pytest.mark.asyncio
async def test_decision_draft_worker_output_error_is_handled_as_needs_replan(memory_store):
    """decision_draft.py DOES catch WorkerOutputError (unlike intake.py above)
    and converts it to needs_replan=True -- proves the "and is handled" half of
    AC-04's evidence contract: a malformed structured-output entry becomes a
    defined, recoverable state update, never a silently-accepted bogus
    PADecision (e.g. a spuriously-approved disposition slipping through
    unvalidated) and never an uncaught crash here."""
    model = ValidatingFakeModel(
        script=[AIMessage(content="drafted")],
        # Missing required `reviewer_summary` -- a genuine ValidationError.
        structured_responses=[{"disposition": "approve", "confidence": 0.9}],
    )
    node = build_decision_draft_node(store=memory_store, model=model)
    request = PARequest(
        member_id="M100001", service_code="72148", diagnosis_codes=["M54.16"],
        requested_units=1, place_of_service="outpatient", provider_npi="1093817465",
        clinical_summary="MRI lumbar spine", missing_fields=[],
    )
    benefit = BenefitResult(covered=True, plan_id="PPO-100", requires_pa=True, network_status="in_network")
    necessity = NecessityAssessment(
        criteria_status="met", policy_id="PA-MRI-LUMBAR", citations=[],
        unmet_requirements=[], confidence=0.9, rationale="documented",
    )
    state = {
        "case_id": "case-ac04-2", "member_id": "M100001", "request": request,
        "benefit": benefit, "necessity": necessity, "retrieved_criteria": [],
    }
    update = await node(state)
    assert update == {"needs_replan": True}
