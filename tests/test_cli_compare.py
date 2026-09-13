"""pac compare: runs a sample through both the multi-agent graph and the
single-agent baseline, prints both decisions. Both paths mocked -- no live
GEMINI_API_KEY / MCP subprocess needed."""

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _full_case import FAKE_MCP_TOOLS, build_clear_cut_case, stub_summarizer  # noqa: E402

from pa_copilot.cli import app  # noqa: E402
from pa_copilot.graph import make_graph  # noqa: E402
from pa_copilot.schemas import PADecision  # noqa: E402

runner = CliRunner()


def test_compare_prints_both_decisions(tmp_path, monkeypatch, fake_embedder):
    monkeypatch.setenv("PA_STATE_DB", str(tmp_path / "state.db"))
    monkeypatch.setenv("PA_MEMORY_DB", str(tmp_path / "mem.db"))
    stub_summarizer(monkeypatch)

    case = build_clear_cut_case()
    monkeypatch.setattr("pa_copilot.cli._real_embedder", lambda settings: fake_embedder)

    async def fake_make_graph(*, store, checkpointer, mcp_tools, model=None, settings=None):
        return await make_graph(
            store=store, checkpointer=checkpointer, mcp_tools=mcp_tools, model=case.model,
        )

    monkeypatch.setattr("pa_copilot.cli.make_graph", fake_make_graph)

    single_decision = PADecision(
        disposition="approve", cited_criteria=[], reviewer_summary="single-agent ok",
        confidence=0.7,
    )

    async def fake_run_single_agent(raw_provider_text, member_id, *, mcp_tools, store, model=None):
        return single_decision

    monkeypatch.setattr("pa_copilot.single_agent.run_single_agent", fake_run_single_agent)

    class _FakeSession:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr("pa_copilot.cli.pa_session", lambda client=None: _FakeSession())
    monkeypatch.setattr("pa_copilot.cli.build_client", lambda: object())

    async def fake_load_pa_tools(client=None, session=None):
        return FAKE_MCP_TOOLS

    monkeypatch.setattr("pa_copilot.cli.load_pa_tools", fake_load_pa_tools)

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

    result = runner.invoke(app, ["compare", str(sample_path)])
    assert result.exit_code == 0, result.output
    assert "multi-agent decision" in result.output
    assert "single-agent decision" in result.output
    assert "single-agent ok" in result.output

    # A second `pac compare` on the same sample must refuse -- its multi-agent
    # thread already has a finished checkpoint, and re-running fresh state on
    # it would append to (not replace) that thread's additive-reducer history,
    # the same hazard pac submit's own guard exists for.
    second = runner.invoke(app, ["compare", str(sample_path)])
    assert second.exit_code == 1
    assert "already has a finished run recorded" in second.output
