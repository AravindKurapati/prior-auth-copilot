"""Task 7 (AC-02/03/10/11): shared clear-cut / ambiguous full-graph case builders.

Not a test module itself (no ``test_*`` functions, so pytest never collects it) --
shared by ``tests/test_ac02_supervisor_routing.py``, ``tests/test_ac03_conditional_routing.py``,
``tests/test_ac11_agentic_rag.py``, and ``scripts/full_case_demo.py`` (which reaches in
via ``sys.path`` the same way it already does for ``_fakes.py`` -- see
``scripts/run_pause_resume_test.py``'s ``_fake_tool_calling_model()``).

Both cases share the same member/service/provider (``sample_request`` fixture's
shape: member ``M100001``, service ``72148``, dx ``M54.16``, provider NPI
``1093817465``) -- the real synthetic corpora's ``criteria_check`` for that
service/dx pair genuinely returns ``status="indeterminate"`` (M54.16 is not one
of PA-MRI-LUMBAR's ``excluded_diagnoses`` -- only M54.5 is, per
``data/synthetic/criteria.json``), so the *fake* MCP tool below intentionally
mirrors that real status rather than inventing a different one.

One shared ``FakeToolCallingModel`` instance drives EVERY model call across the
whole run -- the supervisor's routing turns AND every worker's tool-calling +
structured-output turns (``pa_copilot.graph.make_graph`` wires the same
``model=`` into ``build_supervisor_node`` and every ``build_*_node`` call). Its
``script`` (tool-calling turns) and ``structured_responses`` (structured-output
turns) are two independent FIFO queues, popped in the exact order the real
graph will need them:

Clear-cut turn order (5 supervisor turns; turns 1 and 5 are deterministic
guardrails -- ``hard_route`` -- and consume NEITHER queue):
    1. supervisor (hard route, request=None)      -> intake            [no pop]
    2. intake: script x2, structured x1 (PARequest)
    3. supervisor (LLM)                             -> benefit_check    [structured x1]
    4. benefit_check: script x2, structured x1 (BenefitResult)
    5. supervisor (LLM)                             -> medical_necessity[structured x1]
    6. medical_necessity: script x2, structured x1 (NecessityAssessment)
    7. supervisor (LLM)                             -> decision_draft   [structured x1]
    8. decision_draft: script x1, structured x1 (PADecision)
    9. supervisor (hard route, decision set)        -> FINISH           [no pop]

Ambiguous turn order diverges after medical_necessity: its script has ONE more
tool call (``search_clinical_guidance``, proving AC-11's "agent decides, inside
the compiled graph" for the indeterminate path), its NecessityAssessment is
low-confidence, and the following supervisor turn is scripted to route straight
to ``human_review`` (a normal ``RouterDecision`` choice -- no hop-cap/guardrail
trickery needed, unlike the AC-05 pause/resume tests, since the supervisor's LLM
router can pick ``human_review`` like any other target). ``human_review``'s own
node calls no model at all (just ``interrupt()``), so the run ends there.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from _fakes import FakeToolCallingModel, ai_tool_call
from pa_copilot.schemas import BenefitResult, NecessityAssessment, PADecision, PARequest, RouterDecision

CASE_ID_CLEARCUT = "case-clearcut-full"
CASE_ID_AMBIGUOUS = "case-ambiguous-full"
SESSION_ID = "sess-full-case"
MEMBER_ID = "M100001"
SERVICE_CODE = "72148"
DIAGNOSIS_CODES = ["M54.16"]
PROVIDER_NPI = "1093817465"
RAW_PROVIDER_TEXT = (
    "Requesting prior auth for MRI lumbar spine (72148). Patient has had 8 weeks "
    "of physical therapy and persistent radicular pain. DX M54.16. Ordering "
    f"provider NPI {PROVIDER_NPI}."
)


@tool("provider_lookup")
def fake_provider_lookup(npi: str) -> dict:
    """Look up a provider (fake MCP tool -- full_case_demo.py fakes throughout)."""
    return {"found": True, "npi": npi, "name": "Dr. Pat Vega", "network_status": "in_network"}


@tool("benefit_lookup")
def fake_benefit_lookup(member_id: str, service_code: str) -> dict:
    """Look up benefit coverage (fake MCP tool)."""
    return {
        "found": True, "covered": True, "requires_pa": True,
        "plan_id": "PLAN-GOLD-PPO", "network_status": "in_network",
    }


@tool("criteria_check")
def fake_criteria_check(service_code: str, diagnosis_codes: list) -> dict:
    """Check criteria (fake MCP tool; mirrors the real 72148/M54.16 corpus
    result -- status "indeterminate", policy PA-MRI-LUMBAR -- see module
    docstring)."""
    return {
        "found": True, "policy_id": "PA-MRI-LUMBAR", "status": "indeterminate",
        "required_conditions": [
            "At least 6 weeks of conservative therapy documented.",
            "Persistent or progressive radicular pain, neurologic deficit, or red-flag findings.",
        ],
        "exclusions": [],
    }


FAKE_MCP_TOOLS = [fake_provider_lookup, fake_benefit_lookup, fake_criteria_check]

EXPECTED_REQUEST = PARequest(
    member_id=MEMBER_ID, service_code=SERVICE_CODE, diagnosis_codes=DIAGNOSIS_CODES,
    requested_units=1, place_of_service="outpatient", provider_npi=PROVIDER_NPI,
    clinical_summary="MRI lumbar spine for radicular pain; 8 weeks PT documented",
    missing_fields=[],
)
EXPECTED_BENEFIT = BenefitResult(
    covered=True, plan_id="PLAN-GOLD-PPO", requires_pa=True, network_status="in_network",
)


@dataclass
class ScriptedCase:
    case_id: str
    model: FakeToolCallingModel
    expected_request: PARequest
    expected_benefit: BenefitResult
    expected_necessity: NecessityAssessment
    expected_decision: PADecision | None  # None for the ambiguous case (never runs decision_draft)


def new_initial_state(case_id: str) -> dict:
    from pa_copilot.state import new_case_state  # noqa: PLC0415

    return new_case_state(case_id, SESSION_ID, MEMBER_ID, RAW_PROVIDER_TEXT)


def build_clear_cut_case() -> ScriptedCase:
    """>=2 distinct workers, ends at FINISH via ``decision_draft`` (AC-02/03).
    medical_necessity does NOT call ``search_clinical_guidance`` -- a clearly
    met case needs no narrative interpretation (AC-11's "does not call" half)."""
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
    model = FakeToolCallingModel(
        script=[
            ai_tool_call("provider_lookup", {"npi": PROVIDER_NPI}),
            AIMessage(content="extracted"),
            ai_tool_call("benefit_lookup", {"member_id": MEMBER_ID, "service_code": SERVICE_CODE}),
            AIMessage(content="checked"),
            ai_tool_call("criteria_check", {"service_code": SERVICE_CODE, "diagnosis_codes": DIAGNOSIS_CODES}),
            AIMessage(content="assessed"),
            AIMessage(content="drafted"),
        ],
        structured_responses=[
            EXPECTED_REQUEST,
            RouterDecision(next="benefit_check", rationale="request captured; benefit still unknown"),
            EXPECTED_BENEFIT,
            RouterDecision(next="medical_necessity", rationale="benefit confirmed; necessity still unknown"),
            necessity,
            RouterDecision(next="decision_draft", rationale="necessity clearly met; ready to draft"),
            decision,
        ],
    )
    return ScriptedCase(CASE_ID_CLEARCUT, model, EXPECTED_REQUEST, EXPECTED_BENEFIT, necessity, decision)


def build_ambiguous_case() -> ScriptedCase:
    """Routes to ``human_review`` instead of ``decision_draft`` (AC-03). Forces
    ambiguity via a low-confidence, indeterminate ``NecessityAssessment`` (the
    brief's suggested simplest option) -- the supervisor's own scripted
    ``RouterDecision`` is what actually sends the case to ``human_review``;
    nothing in ``hard_route`` reads ``confidence`` (it only reads
    hops/tool_failures/decision/request), so the "ambiguous" framing is carried
    entirely by the router's semantic choice, exactly as design.md intends.
    medical_necessity DOES call ``search_clinical_guidance`` here (AC-11's
    "calls it" half)."""
    necessity = NecessityAssessment(
        criteria_status="indeterminate", policy_id="PA-MRI-LUMBAR", citations=[],
        unmet_requirements=["conservative-therapy duration not clearly documented"],
        confidence=0.35,
        rationale="mechanical check indeterminate; retrieved guidance inconclusive, low confidence",
    )
    model = FakeToolCallingModel(
        script=[
            ai_tool_call("provider_lookup", {"npi": PROVIDER_NPI}),
            AIMessage(content="extracted"),
            ai_tool_call("benefit_lookup", {"member_id": MEMBER_ID, "service_code": SERVICE_CODE}),
            AIMessage(content="checked"),
            ai_tool_call("criteria_check", {"service_code": SERVICE_CODE, "diagnosis_codes": DIAGNOSIS_CODES}),
            ai_tool_call(
                "search_clinical_guidance",
                {"query": "conservative therapy duration lumbar MRI necessity", "service_code": SERVICE_CODE},
            ),
            AIMessage(content="assessed, low confidence"),
        ],
        structured_responses=[
            EXPECTED_REQUEST,
            RouterDecision(next="benefit_check", rationale="request captured; benefit still unknown"),
            EXPECTED_BENEFIT,
            RouterDecision(next="medical_necessity", rationale="benefit confirmed; necessity still unknown"),
            necessity,
            RouterDecision(
                next="human_review",
                rationale="necessity indeterminate with low confidence; escalate to a human reviewer",
            ),
        ],
    )
    return ScriptedCase(CASE_ID_AMBIGUOUS, model, EXPECTED_REQUEST, EXPECTED_BENEFIT, necessity, None)


def build_stub_summarization_node():
    """A ``SummarizationNode`` that never actually summarizes (thresholds set
    absurdly high) so ``context.summarization.summarize()``'s module-level node
    cache never builds a real ``ChatGoogleGenerativeAI`` (which would raise
    ``DefaultCredentialsError`` with no Google/Gemini credentials configured) --
    same technique as ``tests/test_ac05_checkpointer.py`` /
    ``scripts/run_pause_resume_test.py``. Its own ``FakeToolCallingModel`` is a
    separate instance from a case's ``model`` -- independent queues, never
    shared."""
    from pa_copilot.context.summarization import build_summarization_node  # noqa: PLC0415

    return build_summarization_node(
        model=FakeToolCallingModel(script=[]), max_tokens=100_000, max_tokens_before_summary=100_000,
    )


def stub_summarizer(monkeypatch) -> None:
    """pytest-side installer for :func:`build_stub_summarization_node`."""
    import pa_copilot.context.summarization as summarization_mod  # noqa: PLC0415

    monkeypatch.setattr(summarization_mod, "_node", build_stub_summarization_node())


@contextmanager
def fake_rag_embedder():
    """Point ``search_clinical_guidance``'s embedder at a ``FakeEmbedder`` for a
    run that scripts an actual call to it (the ambiguous case) -- avoids
    downloading the real ``BgeEmbedder`` model. No Chroma index is built, so
    ``index.search(...)`` raises ``RagIndexUnavailable`` internally and the real
    tool returns ``[]`` citations either way -- the fake model's scripted
    ``NecessityAssessment`` does not depend on the tool's return value, only on
    the call happening (or not) at all."""
    from _fakes import FakeEmbedder  # noqa: PLC0415
    from pa_copilot.rag.tool import reset_tool_embedder, set_tool_embedder  # noqa: PLC0415

    set_tool_embedder(FakeEmbedder())
    try:
        yield
    finally:
        reset_tool_embedder()


def make_rag_spy(calls: list):
    """Wrap the REAL ``search_clinical_guidance`` tool so a test can count
    in-graph invocations without faking the tool's own logic -- delegates to
    the real tool's ``.invoke(...)`` (still real retrieval/gating code, just
    with the embedder swapped -- see :func:`fake_rag_embedder`), only recording
    each call's args in ``calls``. Same name (``search_clinical_guidance``) so
    ``ToolNode`` dispatch still matches the scripted ``ai_tool_call``."""
    from pa_copilot.rag.tool import search_clinical_guidance as real_tool  # noqa: PLC0415

    @tool("search_clinical_guidance")
    def _spy(query: str, service_code: str | None = None) -> list:
        """Search the clinical-guidance corpus (call-counting spy over the real tool)."""
        calls.append({"query": query, "service_code": service_code})
        return real_tool.invoke({"query": query, "service_code": service_code})

    return _spy
