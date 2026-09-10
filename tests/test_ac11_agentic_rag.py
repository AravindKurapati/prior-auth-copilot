from pathlib import Path

import pytest

from _fakes import FakeEmbedder
from pa_copilot.config import get_settings
from pa_copilot.rag import index as rag_index
from pa_copilot.rag.tool import (
    reset_tool_embedder,
    search_clinical_guidance,
    set_tool_embedder,
    should_search_guidance,
)


@pytest.fixture
def fake_index(tmp_path, monkeypatch):
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    fake = FakeEmbedder()
    rag_index.build_index(embedder=fake, rebuild=True)
    set_tool_embedder(fake)
    yield
    reset_tool_embedder()


def test_predicate_gates_on_status():
    assert should_search_guidance("indeterminate") is True
    assert should_search_guidance("not_found") is True
    assert should_search_guidance("met", unmet_requirements=["needs 6wk PT"]) is True
    assert should_search_guidance("excluded") is False
    assert should_search_guidance("met") is False


def test_tool_returns_citations_for_psg(fake_index):
    out = search_clinical_guidance.invoke(
        {"query": "attended polysomnography home sleep test screening", "service_code": "95810"}
    )
    assert isinstance(out, list) and out
    assert all(c["source"] == "rag_corpus" for c in out)
    assert any("95810" in c["clause_id"] or "PA-PSG" in c["clause_id"] for c in out)


def test_tool_tolerates_no_hits(fake_index):
    out = search_clinical_guidance.invoke({"query": "zzzzz nonsense tokens qqqq"})
    assert isinstance(out, list)  # possibly empty, never raises


def test_tool_returns_empty_when_index_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "no-index"))
    get_settings.cache_clear()
    set_tool_embedder(FakeEmbedder())
    try:
        out = search_clinical_guidance.invoke({"query": "anything at all"})
    finally:
        reset_tool_embedder()
    assert out == []


def test_default_embedder_is_cached_and_override_restores_it(monkeypatch):
    import pa_copilot.rag.tool as tool

    # monkeypatch.setattr restores both module globals after the test.
    monkeypatch.setattr(tool, "_default_embedder", None)
    monkeypatch.setattr(tool, "_tool_embedder_override", None)

    # No override: the default is built once (lazily — no model download) and reused.
    default = tool._get_tool_embedder()
    assert tool._get_tool_embedder() is default

    # An override wins; reset drops it back to the SAME cached default, not None.
    fake = FakeEmbedder()
    tool.set_tool_embedder(fake)
    assert tool._get_tool_embedder() is fake
    tool.reset_tool_embedder()
    assert tool._get_tool_embedder() is default


def test_ac11_decision_evidence_committed():
    md = Path(get_settings().traces_dir) / "agentic_rag_decision.md"
    assert md.exists()
    body = md.read_text(encoding="utf-8")
    assert "indeterminate" in body and "excluded" in body
    assert "search_clinical_guidance" in body
