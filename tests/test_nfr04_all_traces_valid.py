"""NFR-04 full: every committed traces/*.json file that represents a case run
validates against the trace schema's stable anchor -- not just RunTracer
unit-tested in isolation (tests/test_nfr04_trace_schema.py).

Read every file under traces/ first (per this PR's plan): most committed
*.json case dumps (route_clearcut.json, route_ambiguous.json, run_full_case.json,
reflection_*.json, quarantine_canary.json) are NOT produced by RunTracer.finish()
directly -- scripts/full_case_demo.py's own docstring explains why (a true
RunTracer-shaped trace needs a tracer threaded through every worker node, which
earlier PRs were explicitly barred from adding to graph.py) and names this test
file as the "future schema-unifying pass." The shape they DO all share, by
deliberate convention (same docstring), is `schema_version` under the same key
`tracing.TRACE_SCHEMA_VERSION` uses, plus `case_id` for anything that is a case
trace. mcp_capabilities.json, rag_index_summary.json, and tiered_memory_recall.json
are genuinely NOT case traces (no schema_version, no case_id) and are correctly
excluded, not silently missed.
"""

import json
from pathlib import Path

from pa_copilot.tracing import TRACE_SCHEMA_VERSION

_TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"


def _all_json_traces() -> list[Path]:
    paths = sorted(_TRACES_DIR.glob("*.json"))
    assert paths, "no traces/*.json committed -- expected at least one"
    return paths


def test_every_committed_json_trace_parses_as_utf8_lf_json():
    for path in _all_json_traces():
        raw = path.read_bytes()
        assert b"\r\n" not in raw, f"{path.name} has CRLF line endings"
        json.loads(raw.decode("utf-8"))  # must not raise


def test_every_case_trace_carries_the_stable_schema_version():
    """Any traces/*.json with a `case_id` key is a case-level trace and MUST
    carry `schema_version` matching tracing.TRACE_SCHEMA_VERSION -- the one
    contract scripts/full_case_demo.py and scripts/reflection_demo.py both
    already honor deliberately (see their own docstrings)."""
    checked = 0
    for path in _all_json_traces():
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or "case_id" not in doc:
            continue
        checked += 1
        assert doc.get("schema_version") == TRACE_SCHEMA_VERSION, (
            f"{path.name}: schema_version={doc.get('schema_version')!r}, "
            f"expected {TRACE_SCHEMA_VERSION!r}"
        )
    assert checked, "expected at least one case-level trace (with case_id) under traces/"


def test_non_case_artifacts_are_recognized_and_excluded_deliberately():
    """mcp_capabilities.json / rag_index_summary.json / tiered_memory_recall.json
    are genuinely not case traces (no schema_version, no case_id) -- this test
    documents that exclusion is deliberate, not a gap this file missed."""
    non_case_names = {"mcp_capabilities.json", "rag_index_summary.json", "tiered_memory_recall.json"}
    present = {p.name for p in _all_json_traces()} & non_case_names
    for name in present:
        doc = json.loads((_TRACES_DIR / name).read_text(encoding="utf-8"))
        assert "case_id" not in doc
        assert "schema_version" not in doc
