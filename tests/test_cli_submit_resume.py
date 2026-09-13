"""pac submit / pac resume against a real (fake-model-backed) graph -- no live
GEMINI_API_KEY needed. Verifies the resume path calls Command(resume=...) with
NO manual supervisor_hops/replan_count override (PR6 already resets those in
human_review.py -- this is this task's own proof it isn't reintroducing the
pre-PR6 workaround)."""

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _full_case import FAKE_MCP_TOOLS, build_clear_cut_case, stub_summarizer  # noqa: E402

from pa_copilot.cli import app  # noqa: E402
from pa_copilot.graph import make_graph  # noqa: E402

runner = CliRunner()

_SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


def _patch_graph(monkeypatch, model, store_holder):
    async def fake_build_graph_for_case(store, checkpointer):
        store_holder["store"] = store
        return await make_graph(
            store=store, checkpointer=checkpointer, mcp_tools=FAKE_MCP_TOOLS, model=model
        )

    monkeypatch.setattr("pa_copilot.cli._build_graph_for_case", fake_build_graph_for_case)


def test_submit_writes_trace_and_prints_decision(tmp_path, monkeypatch, fake_embedder):
    monkeypatch.setenv("PA_STATE_DB", str(tmp_path / "state.db"))
    monkeypatch.setenv("PA_MEMORY_DB", str(tmp_path / "mem.db"))
    monkeypatch.setenv("PA_TRACES_DIR", str(tmp_path / "traces"))

    stub_summarizer(monkeypatch)
    case = build_clear_cut_case()
    _patch_graph(monkeypatch, case.model, {})

    # Build a sample file matching this scripted case's own case_id/member_id so
    # the fake model's script (which assumes MEMBER_ID/SERVICE_CODE/etc.) lines up.
    sample = {
        "case_id": case.case_id,
        "session_id": "sess-full-case",
        "member_id": "M100001",
        "raw_provider_text": (
            "Requesting prior auth for MRI lumbar spine (72148). Patient has had 8 "
            "weeks of physical therapy and persistent radicular pain. DX M54.16. "
            "Ordering provider NPI 1093817465."
        ),
        "structured": {
            "service_code": "72148", "diagnosis_codes": ["M54.16"],
            "requested_units": 1, "place_of_service": "outpatient",
            "provider_npi": "1093817465",
        },
    }
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(sample), encoding="utf-8")

    result = runner.invoke(app, ["submit", str(sample_path)])
    assert result.exit_code == 0, result.output
    assert "approve" in result.output

    trace_files = list((tmp_path / "traces").glob("*.json"))
    assert len(trace_files) == 1


def test_resume_does_not_pass_manual_supervisor_hops_override(tmp_path, monkeypatch, fake_embedder):
    """Proves pac resume calls Command(resume=value) with update=None -- the
    pre-PR6 workaround (Command(resume=..., update={"supervisor_hops": 0})) must
    not be reintroduced here; human_review.py already resets on resume."""
    monkeypatch.setenv("PA_STATE_DB", str(tmp_path / "state.db"))
    monkeypatch.setenv("PA_MEMORY_DB", str(tmp_path / "mem.db"))
    stub_summarizer(monkeypatch)

    captured = {}
    from langgraph.graph.state import CompiledStateGraph

    real_ainvoke = CompiledStateGraph.ainvoke

    async def spy_ainvoke(self, arg, config=None, **kw):
        captured["arg"] = arg
        return await real_ainvoke(self, arg, config=config, **kw)

    monkeypatch.setattr(CompiledStateGraph, "ainvoke", spy_ainvoke)

    from _fakes import FakeToolCallingModel  # noqa: PLC0415

    async def fake_build_graph_for_case(store, checkpointer):
        return await make_graph(
            store=store, checkpointer=checkpointer, mcp_tools=FAKE_MCP_TOOLS,
            model=FakeToolCallingModel(script=[], structured_responses=[]),
        )

    monkeypatch.setattr("pa_copilot.cli._build_graph_for_case", fake_build_graph_for_case)

    runner.invoke(app, ["resume", "case-nonexistent", "approved"])
    # A resume against a thread with no prior checkpoint won't produce a real
    # decision, but the assertion this test exists for is about the Command
    # shape passed to ainvoke, not the outcome.
    from langgraph.types import Command

    assert isinstance(captured["arg"], Command)
    assert captured["arg"].resume == "approved"
    assert captured["arg"].update is None
