"""NFR-02: end-to-end from ONE documented command (`pac submit`), with a mocked
LLM, produces a schema-valid PADecision. This is the test the NFR row names
directly -- distinct from tests/test_cli_submit_resume.py's CLI-plumbing test."""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _full_case import FAKE_MCP_TOOLS, build_clear_cut_case, stub_summarizer  # noqa: E402

from pa_copilot.cli import app  # noqa: E402
from pa_copilot.graph import make_graph  # noqa: E402
from pa_copilot.schemas import PADecision  # noqa: E402

runner = CliRunner()


def test_pac_submit_produces_a_schema_valid_decision(tmp_path, monkeypatch, fake_embedder):
    monkeypatch.setenv("PA_STATE_DB", str(tmp_path / "state.db"))
    monkeypatch.setenv("PA_MEMORY_DB", str(tmp_path / "mem.db"))
    monkeypatch.setenv("PA_TRACES_DIR", str(tmp_path / "traces"))
    stub_summarizer(monkeypatch)

    case = build_clear_cut_case()

    @asynccontextmanager
    async def fake_open_graph(store, checkpointer):
        yield await make_graph(
            store=store, checkpointer=checkpointer, mcp_tools=FAKE_MCP_TOOLS, model=case.model,
        )

    monkeypatch.setattr("pa_copilot.cli._open_graph", fake_open_graph)
    monkeypatch.setattr("pa_copilot.cli._real_embedder", lambda settings: fake_embedder)

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

    # Extract the printed JSON decision block (everything after the first line)
    # and validate it against PADecision -- this test's own evidence, distinct
    # from just checking exit_code == 0.
    lines = result.output.strip().splitlines()
    json_text = "\n".join(lines[1:])
    parsed = json.loads(json_text)
    decision = PADecision(**parsed)
    assert decision.disposition == "approve"
