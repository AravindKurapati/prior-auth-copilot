import logging

import pytest

# `tests/` is on sys.path (pytest prepend import mode); re-exported for callers
# that historically did `from tests.test_rag_index import FakeEmbedder`.
from _fakes import FakeEmbedder
from pa_copilot.rag import index as rag_index
from pa_copilot.rag.corpus import load_guidance
from pa_copilot.rag.embedder import EMBED_DIM_BGE_SMALL, BgeEmbedder

__all__ = ["FakeEmbedder"]


@pytest.mark.slow
def test_bge_embedder_shapes():
    e = BgeEmbedder("BAAI/bge-small-en-v1.5")
    v = e.embed_query("polysomnography medical necessity")
    assert len(v) == EMBED_DIM_BGE_SMALL
    d = e.embed_documents(["home sleep apnea test", "attended in-lab study"])
    assert len(d) == 2 and len(d[0]) == EMBED_DIM_BGE_SMALL


def test_build_index_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "chroma"))
    from pa_copilot.config import get_settings

    get_settings.cache_clear()
    fake = FakeEmbedder()
    s1 = rag_index.build_index(embedder=fake, rebuild=True)
    s2 = rag_index.build_index(embedder=fake, rebuild=True)
    assert s1 == s2
    assert s1.chunk_count == len(load_guidance())
    assert s1.doc_count == 6


def test_search_filters_by_service_code(tmp_path, monkeypatch):
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "chroma"))
    from pa_copilot.config import get_settings

    get_settings.cache_clear()
    fake = FakeEmbedder()
    rag_index.build_index(embedder=fake, rebuild=True)
    hits = rag_index.search(
        "home sleep study attended polysomnography", embedder=fake, service_code="95810"
    )
    assert hits and all(h["service_code"] == "95810" for h in hits)
    assert "score" in hits[0]
    assert "chunk_id" in hits[0]


def test_search_without_index_raises_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "empty"))
    from pa_copilot.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(rag_index.RagIndexUnavailable):
        rag_index.search("anything", embedder=FakeEmbedder())


class _Stub384Embedder:
    """384-dim stub — matches bge-small's width, not FakeEmbedder's 64."""

    def embed_documents(self, texts):
        return [[1.0] + [0.0] * 383 for _ in texts]

    def embed_query(self, text):
        return [1.0] + [0.0] * 383


def test_search_degrades_on_dimension_mismatch(tmp_path, monkeypatch):
    """A `.pa_chroma/` built by a 64-dim embedder + a 384-dim query must surface
    as `RagIndexUnavailable`, not a raw `InvalidDimensionException` (I3)."""
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "chroma"))
    from pa_copilot.config import get_settings

    get_settings.cache_clear()
    rag_index.build_index(embedder=FakeEmbedder(), rebuild=True)
    with pytest.raises(rag_index.RagIndexUnavailable):
        rag_index.search("anything at all", embedder=_Stub384Embedder())


def test_search_degrades_when_index_model_changed(tmp_path, monkeypatch):
    """Collection metadata records the build-time model; a config change to a
    different embedding model must degrade, not silently mis-rank (I3)."""
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "chroma"))
    from pa_copilot.config import get_settings

    get_settings.cache_clear()
    rag_index.build_index(embedder=FakeEmbedder(), rebuild=True)
    monkeypatch.setenv("PA_EMBEDDING_MODEL", "some/other-embedding-model")
    get_settings.cache_clear()
    with pytest.raises(rag_index.RagIndexUnavailable):
        rag_index.search("anything at all", embedder=FakeEmbedder())


def test_no_chroma_telemetry_noise(tmp_path, monkeypatch, caplog, capfd):
    """chromadb 0.6.3 emits "Failed to send telemetry event ... capture() takes 1
    positional argument but 3 were given" on every `PersistentClient` /
    collection call. It is a `logging` record (not a bare print) so `caplog` is
    what catches it under pytest; `capfd` additionally guards a future
    print-based regression. Both are checked against the *real* build + search
    path — a module reload never creates a client, so reload-only checks pass
    even with all silencing removed. This fails if the
    `logging.getLogger("chromadb.telemetry")` line in `rag/index.py` is removed
    (verified)."""
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "chroma"))
    from pa_copilot.config import get_settings

    get_settings.cache_clear()
    caplog.set_level(logging.WARNING, logger="chromadb")
    rag_index.build_index(embedder=FakeEmbedder(), rebuild=True)
    rag_index.search("polysomnography", embedder=FakeEmbedder())
    out, err = capfd.readouterr()
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "Failed to send telemetry" not in logged
    assert "Failed to send telemetry" not in err
    assert "Failed to send telemetry" not in out


@pytest.mark.slow
def test_real_model_search_ranks_right_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "chroma"))
    from pa_copilot.config import get_settings

    get_settings.cache_clear()
    from pa_copilot.rag.embedder import BgeEmbedder

    e = BgeEmbedder("BAAI/bge-small-en-v1.5")
    rag_index.build_index(embedder=e, rebuild=True)
    hits = rag_index.search(
        "attended in-lab polysomnography vs home sleep test", embedder=e
    )
    assert hits[0]["policy_id"] == "PA-PSG"
