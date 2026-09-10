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
# one logger subtree fully suppresses it — verified by
# `test_no_chroma_telemetry_noise`, which fails if this is removed. (Layers 1-2
# alone do not: the offending posthog `capture()` call fires before the setting
# binds.) No monkeypatch of chromadb internals is needed.
#
# We deliberately do NOT raise the level of the whole `chromadb` logger: that
# would also gag `local_persistent_hnsw`'s "Number of requested results N is
# greater than number of elements in index M" warning — the exact silent
# retrieval degradation that matters once PR5 adds a `service_code` filter and
# PR7 ships a partial index.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# The specific deprecation chromadb 0.6.3 raises from its own `types.py` under
# pydantic >= 2.11 (`.model_fields` on an instance). Fall back to the parent class
# if this pydantic build predates the dated subclass.
try:  # pragma: no cover - version shim
    from pydantic import PydanticDeprecatedSince211 as _ChromaPydWarning
except Exception:  # pragma: no cover
    from pydantic import PydanticDeprecationWarning as _ChromaPydWarning

# `search_clinical_guidance` has no async impl, so LangGraph's `ToolNode` runs it
# via `run_in_executor` — on threads, concurrently, for parallel tool calls. A
# `warnings.catch_warnings()` context manager mutates the *global*
# `warnings.filters` list, so one thread's `__exit__` could restore filters
# mid-block for another and let chromadb's pydantic deprecation escape into the
# suite's `filterwarnings=["error"]` trap. Python 3.12 has no context-local
# warnings state, so we register these two narrow (category + module scoped)
# ignores once, at import, and never touch `warnings.filters` again.
warnings.filterwarnings("ignore", category=_ChromaPydWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"chromadb\..*")

# `delete_collection` on a missing name raises `ValueError` in chromadb 0.6.3;
# newer builds use `chromadb.errors.NotFoundError`. Anything else (a lock or
# permission error) must propagate, not be swallowed.
_DELETE_MISSING_ERRORS: tuple[type[BaseException], ...] = (ValueError,)
try:  # pragma: no cover - version shim
    from chromadb.errors import NotFoundError as _ChromaNotFoundError

    _DELETE_MISSING_ERRORS = (ValueError, _ChromaNotFoundError)
except Exception:  # pragma: no cover
    pass

# A collection built by a different embedder makes `collection.query` raise this
# raw from chromadb; `search` maps it to `RagIndexUnavailable` so PR5 degrades
# gracefully instead of seeing a hard exception.
try:  # pragma: no cover - version shim
    from chromadb.errors import InvalidDimensionException as _ChromaDimError
except Exception:  # pragma: no cover
    class _ChromaDimError(Exception):
        """Fallback when chromadb does not expose InvalidDimensionException."""

_log = logging.getLogger(__name__)


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
    # NB: no `mkdir` here — `build_index` owns creating the store. `search` must
    # not resurrect a deleted / never-built index dir (M-d); it checks existence
    # and raises `RagIndexUnavailable` before ever calling this.
    return chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
    )


def get_collection(
    client: chromadb.api.ClientAPI, name: str, *, metadata: dict | None = None
):
    return client.get_or_create_collection(name, metadata=metadata or dict(_COSINE))


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
    sha = corpus_sha(guidance_dir)

    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    client = get_client(settings.chroma_dir)
    if rebuild:
        try:
            client.delete_collection(settings.rag_collection)
        except _DELETE_MISSING_ERRORS as exc:
            _log.debug("delete_collection(%s) skipped: %s", settings.rag_collection, exc)
    # Stamp the collection with the embedder + corpus it was built against so a
    # later `search` on a stale / wrong-model `.pa_chroma/` degrades cleanly
    # instead of returning garbage or raising a raw dimension error (I3).
    collection = get_collection(
        client,
        settings.rag_collection,
        metadata={
            **_COSINE,
            "embedding_model": settings.embedding_model,
            "corpus_sha": sha,
        },
    )

    embeddings = embedder.embed_documents([c.text for c in chunks])
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
        corpus_sha=sha,
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
    if the collection is missing, empty, or was built with a different embedder
    or corpus than the one now configured.
    """
    settings, embedder = _resolve(settings, embedder)
    k = k or settings.rag_top_k

    if not Path(settings.chroma_dir).exists():
        raise RagIndexUnavailable(
            f"Chroma store {settings.chroma_dir!r} does not exist — "
            f"run scripts/ingest_rag.py to build it."
        )

    client = get_client(settings.chroma_dir)
    try:
        collection = client.get_collection(settings.rag_collection)
    except Exception as exc:
        raise RagIndexUnavailable(
            f"Chroma collection {settings.rag_collection!r} not found under "
            f"{settings.chroma_dir!r} — run scripts/ingest_rag.py to build it."
        ) from exc
    if collection.count() == 0:
        raise RagIndexUnavailable(
            f"Chroma collection {settings.rag_collection!r} is empty — "
            f"run scripts/ingest_rag.py to build it."
        )

    meta = collection.metadata or {}
    built_model = meta.get("embedding_model")
    if built_model and built_model != settings.embedding_model:
        raise RagIndexUnavailable(
            f"index built with model {built_model!r}; configured for "
            f"{settings.embedding_model!r} — re-run scripts/ingest_rag.py"
        )
    built_sha = meta.get("corpus_sha")
    if built_sha:
        current_sha = corpus_sha(_guidance_dir(settings))
        if built_sha != current_sha:
            raise RagIndexUnavailable(
                f"index built for corpus {built_sha[:12]}; current corpus is "
                f"{current_sha[:12]} — re-run scripts/ingest_rag.py"
            )

    where = {"service_code": service_code} if service_code else None
    try:
        res = collection.query(
            query_embeddings=[embedder.embed_query(query)],
            n_results=k,
            where=where,
            include=["documents", "distances", "metadatas"],
        )
    except _ChromaDimError as exc:
        raise RagIndexUnavailable(
            f"query embedding dimension does not match collection "
            f"{settings.rag_collection!r} — index built with a different embedder; "
            f"re-run scripts/ingest_rag.py"
        ) from exc

    metadatas = res["metadatas"][0]
    distances = res["distances"][0]
    documents = res["documents"][0]
    hits = [
        {
            "chunk_id": (
                f"{md['policy_id']}:"
                f"{corpus.section_slug(str(md['section']))}:"
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
