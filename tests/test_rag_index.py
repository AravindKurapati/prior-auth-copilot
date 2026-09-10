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


def test_search_without_index_raises_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "empty"))
    from pa_copilot.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(rag_index.RagIndexUnavailable):
        rag_index.search("anything", embedder=FakeEmbedder())


def test_no_chroma_telemetry_noise(capfd):
    import importlib

    importlib.reload(rag_index)
    out, err = capfd.readouterr()
    assert "Failed to send telemetry" not in err
