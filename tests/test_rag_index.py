import pytest

from pa_copilot.rag.embedder import EMBED_DIM_BGE_SMALL, BgeEmbedder


@pytest.mark.slow
def test_bge_embedder_shapes():
    e = BgeEmbedder("BAAI/bge-small-en-v1.5")
    v = e.embed_query("polysomnography medical necessity")
    assert len(v) == EMBED_DIM_BGE_SMALL
    d = e.embed_documents(["home sleep apnea test", "attended in-lab study"])
    assert len(d) == 2 and len(d[0]) == EMBED_DIM_BGE_SMALL
