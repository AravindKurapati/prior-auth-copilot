"""Chroma-backed clinical-guidance index: build + filtered similarity search.

`build_index` embeds every guidance clause (via an `Embedder`) into a persistent
Chroma collection; `search` runs a cosine query with an optional `service_code`
metadata filter and returns plain dicts ordered by similarity.

chromadb 0.6.3 emits a "Failed to send telemetry event ... capture() takes 1
positional argument but 3 were given" line to stderr from its posthog client even
with `anonymized_telemetry=False`, because the offending call fires before the
setting binds. The suite runs with "pristine output" + `filterwarnings=error`, so we silence it:
`ANONYMIZED_TELEMETRY` env var set before `import chromadb`, `ChromaSettings(
anonymized_telemetry=False, ...)` on every client, and — the layer that actually
does it — raising the level of the `chromadb.telemetry` logger subtree, since the
line is a `logging.error` record, not a bare `print`.
"""

from __future__ import annotations

import os

# Layer 1: chromadb reads this at import time — must be set before `import chromadb`.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import warnings  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

from pa_copilot.config import Settings, get_settings  # noqa: E402
from pa_copilot.rag import corpus  # noqa: E402
from pa_copilot.rag.embedder import Embedder  # noqa: E402

# Layer 3: the telemetry line ("Failed to send telemetry event ...") is a
# `logging.error` record on `chromadb.telemetry.product.posthog`. Silencing that
# logger subtree fully suppresses it — verified by
# `test_no_chroma_telemetry_noise`, which fails if this is removed. (Layers 1-2
# alone do not: the offending posthog `capture()` call fires before the setting
# binds.) No monkeypatch of chromadb internals is needed.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# The specific deprecation chromadb 0.6.3 raises from its own `types.py` under
# pydantic >= 2.11 (`.model_fields` on an instance). Fall back to the parent class
# if this pydantic build predates the dated subclass.
try:  # pragma: no cover - version shim
    from pydantic import PydanticDeprecatedSince211 as _ChromaPydWarning
except Exception:  # pragma: no cover
    from pydantic import PydanticDeprecationWarning as _ChromaPydWarning

# `delete_collection` on a missing name raises `ValueError` in chromadb 0.6.3;
# newer builds use `chromadb.errors.NotFoundError`. Anything else (a lock or
# permission error) must propagate, not be swallowed.
_DELETE_MISSING_ERRORS: tuple[type[BaseException], ...] = (ValueError,)
try:  # pragma: no cover - version shim
    from chromadb.errors import NotFoundError as _ChromaNotFoundError

    _DELETE_MISSING_ERRORS = (ValueError, _ChromaNotFoundError)
except Exception:  # pragma: no cover
    pass

_log = logging.getLogger(__name__)


@contextmanager
def _quiet_chroma():
    """Scope-ignore the pydantic deprecation chromadb 0.6.3 trips from its own
    `types.py` when its collection objects are touched. The suite runs
    `filterwarnings = ["error"]`; this narrows the ignore to that one third-party
    category (also bounded to `chromadb.*` frames) without weakening the global
    policy — a non-chromadb / non-pydantic deprecation inside the block still trips."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=_ChromaPydWarning)
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module=r"chromadb\..*"
        )
        yield


_COSINE = {"hnsw:space": "cosine"}
_METADATA_KEYS = ("policy_id", "service_code", "section", "doc_title", "clause_index")


class RagIndexUnavailable(RuntimeError):
    """Raised by `search` when the guidance collection is missing or empty."""


@dataclass(frozen=True)
class RagIndexSummary:
    doc_count: int
    chunk_count: int
    embedding_model: str
    corpus_sha: str
    collection: str


def _guidance_dir(settings: Settings) -> Path:
    return Path(settings.synthetic_dir) / "clinical_guidance"


def corpus_sha(guidance_dir: str | Path) -> str:
    """sha1 over the guidance `*.md` file bytes, in sorted-filename order."""
    h = hashlib.sha1()
    for path in sorted(Path(guidance_dir).glob("*.md")):
        h.update(path.read_bytes())
    return h.hexdigest()


def get_client(chroma_dir: str | Path) -> chromadb.api.ClientAPI:
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    with _quiet_chroma():
        return chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
        )


def get_collection(client: chromadb.api.ClientAPI, name: str):
    with _quiet_chroma():
        return client.get_or_create_collection(name, metadata=_COSINE)


def _resolve(settings: Settings | None, embedder: Embedder | None) -> tuple[Settings, Embedder]:
    settings = settings or get_settings()
    if embedder is None:
        from pa_copilot.rag.embedder import BgeEmbedder

        embedder = BgeEmbedder(settings.embedding_model, query_prefix=settings.rag_query_prefix)
    return settings, embedder


def build_index(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    rebuild: bool = True,
) -> RagIndexSummary:
    """Embed every guidance clause into the Chroma collection and return a summary.

    `rebuild=True` drops the existing collection first (via `delete_collection` —
    never an `rmtree`, which deadlocks against a live client on Windows). Also
    writes `<chroma_dir>/rag_manifest.json` (timestamped, not committed).
    """
    settings, embedder = _resolve(settings, embedder)
    guidance_dir = _guidance_dir(settings)
    chunks = corpus.load_guidance(guidance_dir)

    client = get_client(settings.chroma_dir)
    if rebuild:
        try:
            with _quiet_chroma():
                client.delete_collection(settings.rag_collection)
        except _DELETE_MISSING_ERRORS as exc:
            _log.debug("delete_collection(%s) skipped: %s", settings.rag_collection, exc)
    collection = get_collection(client, settings.rag_collection)

    embeddings = embedder.embed_documents([c.text for c in chunks])
    with _quiet_chroma():
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{k: getattr(c, k) for k in _METADATA_KEYS} for c in chunks],
        )

    summary = RagIndexSummary(
        doc_count=len({c.policy_id for c in chunks}),
        chunk_count=len(chunks),
        embedding_model=settings.embedding_model,
        corpus_sha=corpus_sha(guidance_dir),
        collection=settings.rag_collection,
    )

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "chroma_dir": str(settings.chroma_dir),
        **{k: getattr(summary, k) for k in ("doc_count", "chunk_count", "embedding_model",
                                            "corpus_sha", "collection")},
    }
    manifest_path = Path(settings.chroma_dir) / "rag_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")

    return summary


def search(
    query: str,
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    k: int | None = None,
    service_code: str | None = None,
) -> list[dict]:
    """Cosine similarity search over the guidance index.

    Returns dicts keyed `chunk_id, policy_id, service_code, section, doc_title,
    clause_index, text, score` (``score = 1 - distance``), ordered by score
    descending. `chunk_id` is rebuilt to match `corpus.Chunk.chunk_id`
    (``{policy_id}:{section-slug}:{clause_index}``) so callers need not re-slugify.
    Filters on `service_code` metadata when given. Raises `RagIndexUnavailable`
    if the collection is missing or empty.
    """
    settings, embedder = _resolve(settings, embedder)
    k = k or settings.rag_top_k

    client = get_client(settings.chroma_dir)
    try:
        with _quiet_chroma():
            collection = client.get_collection(settings.rag_collection)
    except Exception as exc:
        raise RagIndexUnavailable(
            f"Chroma collection {settings.rag_collection!r} not found under "
            f"{settings.chroma_dir!r} — run scripts/ingest_rag.py to build it."
        ) from exc
    with _quiet_chroma():
        empty = collection.count() == 0
    if empty:
        raise RagIndexUnavailable(
            f"Chroma collection {settings.rag_collection!r} is empty — "
            f"run scripts/ingest_rag.py to build it."
        )

    where = {"service_code": service_code} if service_code else None
    with _quiet_chroma():
        res = collection.query(
            query_embeddings=[embedder.embed_query(query)],
            n_results=k,
            where=where,
            include=["documents", "distances", "metadatas"],
        )

    metadatas = res["metadatas"][0]
    distances = res["distances"][0]
    documents = res["documents"][0]
    hits = [
        {
            "chunk_id": (
                f"{md['policy_id']}:"
                f"{str(md['section']).lower().replace(' ', '-')}:"
                f"{md['clause_index']}"
            ),
            "policy_id": md["policy_id"],
            "service_code": md["service_code"],
            "section": md["section"],
            "doc_title": md["doc_title"],
            "clause_index": md["clause_index"],
            "text": doc,
            "score": 1.0 - dist,
        }
        for md, dist, doc in zip(metadatas, distances, documents)
    ]
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits
