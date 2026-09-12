"""NFR-08: a long synthetic thread triggers summarization; state["context"]
running summary populated; traces/context_before_after.md shows token reduction."""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from summarization_demo import run_demo  # noqa: E402


@pytest.mark.asyncio
async def test_nfr08_long_thread_triggers_summary_and_reduces_tokens(tmp_trace_dir: Path):
    result = await run_demo(trace_dir=tmp_trace_dir)
    assert result["context"]["running_summary"] is not None
    assert result["tokens_after"] < result["tokens_before"]
    out = tmp_trace_dir / "context_before_after.md"
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "NFR-08" in text
    assert str(result["tokens_before"]) in text and str(result["tokens_after"]) in text
