import logging
import zlib

import pytest

from pa_copilot.rag import index as rag_index
from pa_copilot.rag.corpus import load_guidance
from pa_copilot.rag.embedder import EMBED_DIM_BGE_SMALL, BgeEmbedder


@pytest.mark.slow
def test_bge_embedder_shapes():
    e = BgeEmbedder("BAAI/bge-small-en-v1.5")
    v = e.embed_query("polysomnography medical necessity")
    assert len(v) == EMBED_DIM_BGE_SMALL
    d = e.embed_documents(["home sleep apnea test", "attended in-lab study"])
    assert len(d) == 2 and len(d[0]) == EMBED_DIM_BGE_SMALL


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder — no model download.

    Uses ``zlib.crc32`` for token bucketing because builtin ``hash()`` on strings
    is per-process randomized (PYTHONHASHSEED), which would make the index
    non-reproducible across runs.
    """

    DIM = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for tok in text.lower().split():
            v[zlib.crc32(tok.encode()) % self.DIM] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


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
