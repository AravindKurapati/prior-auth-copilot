"""decision_draft worker: synthesizes benefit + necessity into a PADecision
(no tools — pure synthesis), self-critiques every cited quote against
retrieved_criteria, and records a critical-importance memory on denial via a
direct PolicyStore.put (LangMem's generic manage_memory tool hardcodes
routine importance and cannot express this)."""

import asyncio

import pytest
from langchain_core.messages import AIMessage

from _fakes import FakeToolCallingModel
from pa_copilot.agents.decision_draft import build_decision_draft_node
from pa_copilot.config import get_settings
from pa_copilot.schemas import BenefitResult, CriteriaCitation, NecessityAssessment, PADecision, PARequest


def _state(retrieved_criteria) -> dict:
    request = PARequest(member_id="M100001", service_code="72148", diagnosis_codes=["M54.16"],
                         requested_units=1, place_of_service="outpatient", provider_npi="1093817465",
                         clinical_summary="MRI lumbar spine", missing_fields=[])
    benefit = BenefitResult(covered=True, plan_id="PPO-100", requires_pa=True, network_status="in_network")
    necessity = NecessityAssessment(criteria_status="met", policy_id="PA-MRI-LUMBAR",
                                     citations=retrieved_criteria, unmet_requirements=[],
                                     confidence=0.9, rationale="documented")
    return {"case_id": "case-0001", "member_id": "M100001", "request": request, "benefit": benefit,
            "necessity": necessity, "retrieved_criteria": retrieved_criteria}


@pytest.mark.asyncio
async def test_decision_draft_happy_path_approve(memory_store):
    citation = CriteriaCitation(source="rag_corpus", clause_id="c1", quote="8 weeks PT documented", relevance="high")
    decision = PADecision(disposition="approve", cited_criteria=[citation],
                           reviewer_summary="criteria met", confidence=0.85, human_review_required=True)
    model = FakeToolCallingModel(script=[AIMessage(content="drafted")], structured_responses=[decision])
    node = build_decision_draft_node(store=memory_store, model=model)
    update = await node(_state([citation]))
    assert update["decision"] == decision


@pytest.mark.asyncio
async def test_decision_draft_citation_mismatch_sets_needs_replan(memory_store):
    real_citation = CriteriaCitation(source="rag_corpus", clause_id="c1", quote="8 weeks PT documented", relevance="high")
    hallucinated = CriteriaCitation(source="rag_corpus", clause_id="c2", quote="a quote that was never retrieved", relevance="high")
    decision = PADecision(disposition="approve", cited_criteria=[hallucinated],
                           reviewer_summary="criteria met", confidence=0.85)
    model = FakeToolCallingModel(script=[AIMessage(content="drafted")], structured_responses=[decision])
    node = build_decision_draft_node(store=memory_store, model=model)
    update = await node(_state([real_citation]))
    assert update.get("needs_replan") is True
    assert "decision" not in update


@pytest.mark.asyncio
async def test_decision_draft_deny_writes_critical_memory(memory_store):
    decision = PADecision(disposition="deny", cited_criteria=[], reviewer_summary="not met",
                           confidence=0.7, human_review_required=True)
    model = FakeToolCallingModel(script=[AIMessage(content="drafted")], structured_responses=[decision])
    node = build_decision_draft_node(store=memory_store, model=model)
    await node(_state([]))
    items = memory_store.search(("pa", "member", "M100001"), limit=10)
    assert any(i.value.get("importance") == "critical" for i in items)


@pytest.mark.asyncio
async def test_decision_draft_defensive_fallback_when_necessity_missing(memory_store):
    """PR5b final-review Fix B, part 2: hard_route's new necessity-None ->
    medical_necessity guardrail is what actually prevents this in practice,
    but this node must not itself crash dereferencing a None necessity if it
    is ever reached out of order — belt-and-suspenders, not a substitute for
    the routing fix. model=None here: if the node tried to proceed to
    run_worker_react instead of returning immediately, it would blow up
    calling get_agent_model() with no GEMINI_API_KEY configured, so reaching
    the assertion at all proves the early return fired."""
    state = _state([])
    state["necessity"] = None
    node = build_decision_draft_node(store=memory_store, model=None)
    update = await node(state)
    assert update == {"needs_replan": True}


@pytest.mark.asyncio
async def test_decision_draft_fast_path_when_not_covered(memory_store):
    """PR8: design.md §3.3's short-circuit — benefit.covered=False is
    determinative on its own, so necessity=None is a valid input, not a
    defensive-fallback trigger."""
    state = _state([])
    state["benefit"] = BenefitResult(
        covered=False, plan_id="PPO-100", requires_pa=True, network_status="in_network"
    )
    state["necessity"] = None
    decision = PADecision(disposition="deny", cited_criteria=[], reviewer_summary="not covered",
                           confidence=0.95, human_review_required=True)
    model = FakeToolCallingModel(script=[AIMessage(content="drafted")], structured_responses=[decision])
    node = build_decision_draft_node(store=memory_store, model=model)
    update = await node(state)
    assert update["decision"] == decision


@pytest.mark.asyncio
async def test_decision_draft_fast_path_when_no_pa_required(memory_store):
    state = _state([])
    state["benefit"] = BenefitResult(
        covered=True, plan_id="PPO-100", requires_pa=False, network_status="in_network"
    )
    state["necessity"] = None
    decision = PADecision(disposition="approve", cited_criteria=[], reviewer_summary="no PA required",
                           confidence=0.95, human_review_required=True)
    model = FakeToolCallingModel(script=[AIMessage(content="drafted")], structured_responses=[decision])
    node = build_decision_draft_node(store=memory_store, model=model)
    update = await node(state)
    assert update["decision"] == decision


@pytest.mark.asyncio
async def test_decision_draft_model_timeout_degrades_instead_of_raising(memory_store, monkeypatch):
    """Final whole-branch review finding (PR6): decision_draft was the only one
    of the four workers not catching WorkerToolError, so a slow/hung model call
    (run_worker_react_resilient's worker_timeout_seconds wraps the WHOLE turn,
    model call included, even though this worker has zero tools) escaped
    graph.ainvoke() uncaught -- contradicting NFR-07's "never an unhandled
    exception" claim. This test hangs the MODEL itself (not a tool, since
    decision_draft binds none) to prove the fix."""
    monkeypatch.setenv("PA_WORKER_TIMEOUT_SECONDS", "0.05")
    get_settings.cache_clear()  # memory_store fixture already primed the
    # lru_cache with the OLD env value during its own setup (same footgun as
    # test_nfr07_degradation.py's timeout test).

    class HangingModel(FakeToolCallingModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            await asyncio.sleep(999)

    model = HangingModel(script=[AIMessage(content="drafted")], structured_responses=[])
    node = build_decision_draft_node(store=memory_store, model=model)

    update = await node(_state([]))  # must not raise

    assert update["needs_replan"] is True
    assert update["tool_failures"][0].tool == "_worker_turn_"
