"""Adapt any duck-typed embedder to the LangChain `Embeddings` interface that
`SqliteStore`'s index config wants. The real model is bge-small (reused from
`rag/`); tests pass `tests/_fakes.py::FakeEmbedder` through `as_embeddings`."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from pa_copilot.config import get_settings
from pa_copilot.rag.embedder import BgeEmbedder

_default_bge: BgeEmbedder | None = None


def _shared_bge(model_name: str, query_prefix: str) -> BgeEmbedder:
    global _default_bge
    if _default_bge is None or _default_bge.model_name != model_name:
        _default_bge = BgeEmbedder(model_name, query_prefix=query_prefix)
    return _default_bge


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str | None = None, query_prefix: str | None = None) -> None:
        s = get_settings()
        self._bge = _shared_bge(
            model_name or s.embedding_model,
            query_prefix if query_prefix is not None else s.rag_query_prefix,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._bge.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._bge.embed_query(text)


class _DuckEmbeddings(Embeddings):
    def __init__(self, inner: object) -> None:
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


def as_embeddings(obj: object) -> Embeddings:
    if isinstance(obj, Embeddings):
        return obj
    if hasattr(obj, "embed_documents") and hasattr(obj, "embed_query"):
        return _DuckEmbeddings(obj)
    raise TypeError(f"{obj!r} is not an Embeddings and has no embed_documents/embed_query")


def embedding_dims(obj: object) -> int:
    return len(as_embeddings(obj).embed_query("dimension probe"))
