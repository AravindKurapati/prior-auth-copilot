import json
from pathlib import Path

import pytest
from pydantic import ValidationError

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
