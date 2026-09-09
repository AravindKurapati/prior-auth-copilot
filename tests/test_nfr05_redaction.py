import json

from pa_copilot.tracing import RunTracer, load_trace, redact


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
