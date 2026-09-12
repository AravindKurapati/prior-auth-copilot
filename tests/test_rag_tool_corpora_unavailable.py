"""PR3 park, closed in PR5b: search_clinical_guidance's corrective-rewrite path
calls _service_name() -> list_policies() -> load_corpora(), which can raise
CorporaUnavailable — a different exception than the tool's own
RagIndexUnavailable guard. Both must degrade to [] uniformly."""

from pa_copilot.mcp_server.data_access import CorporaUnavailable
from pa_copilot.rag import tool as T


def test_corpora_unavailable_during_rewrite_degrades_to_empty(monkeypatch, fake_embedder):
    T.set_tool_embedder(fake_embedder)

    # Force the corrective-rewrite path by returning empty from first search.
    monkeypatch.setattr(T.index, "search", lambda *a, **kw: [])

    def raise_corpora_unavailable(service_code):
        raise CorporaUnavailable("synthetic corpora missing")

    monkeypatch.setattr(T, "_service_name", raise_corpora_unavailable)

    result = T.search_clinical_guidance.invoke({"query": "anything", "service_code": "72148"})
    assert result == []
    T.reset_tool_embedder()
