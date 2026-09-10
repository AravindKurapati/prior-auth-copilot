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
    assert should_search_guidance("not_met") is True
    assert should_search_guidance("met", unmet_requirements=["needs 6wk PT"]) is True
    assert should_search_guidance("excluded") is False
    assert should_search_guidance("met") is False


def _hit(score: float, section: str = "Indications", idx: int = 0) -> dict:
    return {
        "chunk_id": f"PA-PSG:{section.lower().replace(' ', '-')}:{idx}",
        "policy_id": "PA-PSG",
        "service_code": "95810",
        "section": section,
        "doc_title": "Attended Polysomnography - Medical Necessity",
        "clause_index": idx,
        "text": f"clause {section} {idx}",
        "score": score,
    }


def _patch_search(monkeypatch, *result_sets):
    """Make ``index.search`` return each list in turn; record the queries."""
    import pa_copilot.rag.tool as tool

    calls: list[str] = []
    pending = list(result_sets)

    def fake_search(query, *, service_code=None, embedder=None):
        calls.append(query)
        return pending.pop(0) if pending else []

    monkeypatch.setattr(tool.index, "search", fake_search)
    tool.set_tool_embedder(FakeEmbedder())
    return calls


def test_tool_filters_out_sub_threshold_hits(monkeypatch):
    """A hit below rag_min_score (0.30) is dropped; a clearing hit means no rewrite."""
    calls = _patch_search(monkeypatch, [_hit(0.42), _hit(0.11, "Exclusions")])
    try:
        out = search_clinical_guidance.invoke({"query": "polysomnography"})
    finally:
        import pa_copilot.rag.tool as tool

        tool.reset_tool_embedder()
    assert len(calls) == 1  # cleared on first search -> no corrective rewrite
    assert [c["clause_id"] for c in out] == ["PA-PSG:indications:0"]


def test_tool_rewrite_returns_stronger_rewritten_set(monkeypatch):
    """First search all-weak -> rewrite -> rewritten set clears the bar."""
    calls = _patch_search(
        monkeypatch,
        [_hit(0.20), _hit(0.10, "Exclusions")],
        [_hit(0.55), _hit(0.28, "Exclusions")],
    )
    try:
        out = search_clinical_guidance.invoke({"query": "polysomnography"})
    finally:
        import pa_copilot.rag.tool as tool

        tool.reset_tool_embedder()
    assert len(calls) == 2
    assert [c["clause_id"] for c in out] == ["PA-PSG:indications:0"]
    assert out[0]["relevance"].endswith("score 0.55")


def test_tool_rewrite_keeps_better_original_set_not_weaker_rewrite(monkeypatch):
    """Originals rank better than the rewrite but still miss 0.30 -> [] (the
    weaker rewritten hits are NOT returned at a lower bar)."""
    calls = _patch_search(
        monkeypatch,
        [_hit(0.29), _hit(0.22, "Exclusions")],
        [_hit(0.24), _hit(0.10, "Exclusions")],
    )
    try:
        out = search_clinical_guidance.invoke({"query": "polysomnography"})
    finally:
        import pa_copilot.rag.tool as tool

        tool.reset_tool_embedder()
    assert len(calls) == 2
    assert out == []


def test_tool_all_weak_after_rewrite_returns_empty(monkeypatch):
    calls = _patch_search(
        monkeypatch,
        [_hit(0.22), _hit(0.10, "Exclusions")],
        [_hit(0.25), _hit(0.05, "Exclusions")],
    )
    try:
        out = search_clinical_guidance.invoke({"query": "polysomnography"})
    finally:
        import pa_copilot.rag.tool as tool

        tool.reset_tool_embedder()
    assert len(calls) == 2
    assert out == []


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
