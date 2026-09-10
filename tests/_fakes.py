"""Shared test doubles.

`tests/` is not a package (no `__init__.py`), so cross-file sharing goes through
this module, which pytest's ``prepend`` import mode puts on ``sys.path`` alongside
the test files. Keep it dependency-light — plain classes, no pytest fixtures.
"""

from __future__ import annotations

import zlib


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
