"""NFR-04: structured JSON traces are robust and schema-stable.

Covers `RunTracer` serialization hardening: non-str dict keys (PR4's tuple
namespaces), degraded-trace fallback, `session_id`, and UTF-8 / LF on-disk bytes.
"""

import json

from pa_copilot.tracing import TRACE_SCHEMA_VERSION, RunTracer, load_trace


def test_trace_doc_has_stable_shape(tmp_trace_dir):
    t = RunTracer(tmp_trace_dir, case_id="case-shape", redact_values=[], session_id="sess-1")
    t.event("intake", "worker_output", {"ok": True})
    path = t.finish(decision={"disposition": "approve"})

    doc = load_trace(path)
    assert doc["schema_version"] == TRACE_SCHEMA_VERSION == "1"
    assert doc["case_id"] == "case-shape"
    assert doc["session_id"] == "sess-1"
    assert doc["started_at"] and doc["finished_at"]
    assert doc["events"][0]["node"] == "intake"
    assert doc["decision"]["disposition"] == "approve"


def test_finish_handles_tuple_namespace_keys(tmp_trace_dir):
    """PR4's first memory-write trace carries `SqliteStore` tuple namespaces as
    dict keys — `finish()` must not raise, and must still redact inside the key."""
    t = RunTracer(tmp_trace_dir, case_id="case-tuple", redact_values=[])
    t.event("memory", "store_write", {("pa", "member", "M100001"): "determination record"})
    path = t.finish()

    doc = load_trace(path)  # no exception
    payload = doc["events"][0]["payload"]
    key = next(iter(payload))
    assert isinstance(key, str)
    assert "member" in key
    assert "M100001" not in key
    assert "<redacted>" in key


def test_finish_writes_degraded_trace_on_serialization_failure(tmp_trace_dir):
    class Boom:
        def __repr__(self) -> str:  # pragma: no cover - trivial
            raise RuntimeError("cannot repr")

    t = RunTracer(tmp_trace_dir, case_id="case-degraded", redact_values=[])
    # A key that is neither primitive nor safely str()-able forces the fallback.
    t.event("intake", "worker_output", {Boom(): "x"})
    path = t.finish()

    doc = load_trace(path)
    assert doc["schema_version"] == "1"
    assert doc["case_id"] == "case-degraded"
    assert "error" in doc
    assert doc["events_count"] == 1


def test_trace_file_is_utf8_lf(tmp_trace_dir):
    t = RunTracer(tmp_trace_dir, case_id="case-bytes", redact_values=[])
    t.event("intake", "note", {"text": "café — dash"})
    path = t.finish()

    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.decode("utf-8")  # valid utf-8
    assert json.loads(raw)
