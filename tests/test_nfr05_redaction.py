import json
from pathlib import Path

from pa_copilot.schemas import PARequest
from pa_copilot.tracing import RunTracer, default_redact_values, load_trace, redact

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_redact_scrubs_exact_secrets_and_member_ids():
    doc = {"member_id": "M100001", "note": "call M100001 re: Jane Roe", "n": 3}
    out = redact(doc, secrets=["Jane Roe"])
    assert "M100001" not in json.dumps(out)
    assert "Jane Roe" not in json.dumps(out)
    assert out["n"] == 3


def test_tracer_writes_redacted_json(tmp_trace_dir):
    t = RunTracer(tmp_trace_dir, case_id="case-0001", redact_values=["M100001", "Jane Roe"])
    t.event("intake", "worker_output", {"member_id": "M100001", "patient": "Jane Roe"})
    path = t.finish(decision={"disposition": "approve", "member_id": "M100001"})

    raw = path.read_text()
    assert "M100001" not in raw
    assert "Jane Roe" not in raw

    doc = load_trace(path)
    assert doc["schema_version"] == "1"
    assert doc["case_id"] == "case-0001"
    assert doc["events"][0]["node"] == "intake"
    assert doc["decision"]["disposition"] == "approve"


def test_tracer_redacts_non_primitive_object_stringified_values(tmp_trace_dir):
    """Verify that non-primitive objects (Pydantic models, etc.) are normalized to JSON
    before redaction, so their str() representation doesn't bypass redaction."""
    t = RunTracer(tmp_trace_dir, case_id="case-0002", redact_values=["M100002"])
    # Pass a Pydantic model with member_id — its str() would contain the member_id
    req = PARequest(
        member_id="M100002",
        service_code="72148",
        diagnosis_codes=["M54.16"],
        requested_units=1,
        place_of_service="outpatient",
        provider_npi="1093817465",
        clinical_summary="test",
    )
    t.event("intake", "worker_output", {"request": req})
    path = t.finish()

    raw = path.read_text()
    # Verify member_id from the Pydantic object is redacted
    assert "M100002" not in raw
    assert "<redacted>" in raw


def test_default_redact_values_covers_provider_names_and_member_ids():
    values = default_redact_values()
    assert "Dr. Pat Vega" in values
    assert "M100001" in values


def test_tracer_redacts_synthetic_names_with_empty_caller_list(tmp_trace_dir):
    """NFR-05: synthetic provider names must not leak even when the caller passes
    `redact_values=[]` — the tracer unions in `default_redact_values()`."""
    providers = json.loads(
        (_REPO_ROOT / "data" / "synthetic" / "providers.json").read_text(encoding="utf-8")
    )
    a_name = next(iter(providers.values()))["name"]

    t = RunTracer(tmp_trace_dir, case_id="case-nfr05", redact_values=[])
    t.event("intake", "worker_output", {"name": a_name, "member_id": "M100001"})
    path = t.finish()

    raw = path.read_text(encoding="utf-8")
    assert a_name not in raw
    assert "M100001" not in raw


def test_redact_scrubs_dict_keys():
    """Verify that dict keys containing member IDs or secrets are redacted."""
    doc = {"M100003": "some_value", "Jane Roe": "other_value"}
    out = redact(doc, secrets=["Jane Roe"])
    # Keys should be redacted
    assert "M100003" not in json.dumps(out)
    assert "Jane Roe" not in json.dumps(out)
    assert "<redacted>" in str(list(out.keys()))
