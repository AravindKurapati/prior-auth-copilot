"""Local text embeddings behind a small protocol.

`Embedder` is the structural interface the index (Task 3) depends on; `BgeEmbedder`
is the real implementation, a lazily-loaded CPU `sentence-transformers` model so
importing this module stays cheap and offline-safe.
"""

from __future__ import annotations

from typing import Protocol

EMBED_DIM_BGE_SMALL = 384


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _as_float_lists(vectors) -> list[list[float]]:
    tolist = getattr(vectors, "tolist", None)
    rows = tolist() if callable(tolist) else vectors
    return [[float(x) for x in row] for row in rows]


class BgeEmbedder:
    """`sentence-transformers` embedder pinned to CPU with normalized vectors.

    `model_name` is a HuggingFace id (e.g. ``BAAI/bge-small-en-v1.5``). BGE retrieval
    expects a short instruction prefixed to *queries* only; pass it as `query_prefix`.
    """

    def __init__(self, model_name: str, query_prefix: str = "") -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._get_model().encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return _as_float_lists(vectors)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"{self.query_prefix}{text}"])[0]
