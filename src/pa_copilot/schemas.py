from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RouteTarget = Literal[
    "intake", "benefit_check", "medical_necessity", "decision_draft", "human_review", "FINISH"
]


class CriteriaCitation(BaseModel):
    source: Literal["mcp_resource", "rag_corpus"]
    clause_id: str
    quote: str
    relevance: str


class PARequest(BaseModel):
    member_id: str
    service_code: str
    diagnosis_codes: list[str]
    requested_units: int = Field(ge=1)
    place_of_service: str
    provider_npi: str
    clinical_summary: str
    missing_fields: list[str] = Field(default_factory=list)


class BenefitResult(BaseModel):
    covered: bool
    plan_id: str
    requires_pa: bool
    network_status: str
    notes: str = ""


class NecessityAssessment(BaseModel):
    criteria_status: Literal["met", "not_met", "indeterminate"]
    policy_id: str | None = None
    citations: list[CriteriaCitation] = Field(default_factory=list)
    unmet_requirements: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class PADecision(BaseModel):
    disposition: Literal["approve", "deny", "refer_clinical_review"]
    cited_criteria: list[CriteriaCitation] = Field(default_factory=list)
    reviewer_summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    human_review_required: bool = True


class RouterDecision(BaseModel):
    next: RouteTarget
    rationale: str


class ToolFailure(BaseModel):
    tool: str
    error: str
    attempt: int
    ts: str


class RouteStep(BaseModel):
    from_node: str
    to_node: str
    reason: str
    ts: str
