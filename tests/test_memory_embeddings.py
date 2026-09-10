"""Embeddings adapter — duck-type and LangChain interface."""

import pytest

from pa_copilot.memory import embeddings as E


def test_as_embeddings_wraps_a_duck(fake_embedder):
    emb = E.as_embeddings(fake_embedder)
    assert hasattr(emb, "embed_documents") and hasattr(emb, "embed_query")
    v = emb.embed_query("hello world")
    assert isinstance(v, list) and len(v) == fake_embedder.DIM


def test_as_embeddings_passthrough(fake_embedder):
    wrapped = E.as_embeddings(fake_embedder)
    assert E.as_embeddings(wrapped) is wrapped


def test_as_embeddings_rejects_junk():
    with pytest.raises(TypeError):
        E.as_embeddings(object())


def test_embedding_dims(fake_embedder):
    assert E.embedding_dims(fake_embedder) == 64


@pytest.mark.slow
def test_local_embeddings_real_bge_dims():
    assert E.embedding_dims(E.LocalEmbeddings()) == 384
