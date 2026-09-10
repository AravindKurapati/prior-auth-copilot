"""Agentic-RAG surface: the decision predicate + the retrieval `@tool`.

Retrieval over the clinical-guidance corpus is *inside* the agent loop, not a
fixed pipeline stage: PR5's ``medical_necessity`` worker first asks
``should_search_guidance(...)`` and only then invokes ``search_clinical_guidance``.
This module owns both halves so the "when" and the "how" of retrieval stay
together.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from data.synthetic.generators import SERVICES
from pa_copilot.config import get_settings
from pa_copilot.rag import index
from pa_copilot.rag.embedder import Embedder
from pa_copilot.rag.index import RagIndexUnavailable
from pa_copilot.schemas import CriteriaCitation

_log = logging.getLogger(__name__)

_REWRITE_SUFFIX = " medical necessity criteria indications"

# Module-level embedder override. ``None`` -> lazily build the real ``BgeEmbedder``
# per call; tests set a ``FakeEmbedder`` here via ``set_tool_embedder`` so no model
# is downloaded.
_tool_embedder: Embedder | None = None


def set_tool_embedder(embedder: Embedder) -> None:
    """Force ``search_clinical_guidance`` to use ``embedder`` (test seam)."""
    global _tool_embedder
    _tool_embedder = embedder


def reset_tool_embedder() -> None:
    """Clear any override set by :func:`set_tool_embedder`."""
    global _tool_embedder
    _tool_embedder = None


def _get_tool_embedder() -> Embedder:
    if _tool_embedder is not None:
        return _tool_embedder
    # Lazy: never construct the sentence-transformers model at import time.
    from pa_copilot.rag.embedder import BgeEmbedder

    return BgeEmbedder(get_settings().embedding_model)


def _service_name(service_code: str | None) -> str | None:
    if not service_code:
        return None
    for svc in SERVICES:
        if svc["service_code"] == service_code:
            return svc["name"]
    return None


def should_search_guidance(
    criteria_status: str,
    *,
    unmet_requirements: list[str] | None = None,
) -> bool:
    """Predicate PR5's ``medical_necessity`` worker calls before retrieval.

    Retrieval stays *inside* the agent loop rather than being a fixed pipeline
    step: the worker only reaches for the narrative guidance when the mechanical
    MCP ``criteria_check`` cannot settle the question.

    Returns ``True`` when:

    - ``criteria_status == "indeterminate"`` — policy exists but only a human /
      the narrative can verify the conditions;
    - ``criteria_status == "not_found"`` — no structured policy at all, the
      narrative is all we have;
    - ``unmet_requirements`` is non-empty — checklist gaps that need
      interpretation against the guidance.

    Returns ``False`` for ``"excluded"`` (a mechanical deny signal — no narrative
    needed) and for a clear ``"met"``.
    """
    if criteria_status == "excluded":
        return False
    if criteria_status in ("indeterminate", "not_found"):
        return True
    return bool(unmet_requirements)


@tool
def search_clinical_guidance(query: str, service_code: str | None = None) -> list[dict]:
    """Search the clinical-guidance corpus for medical-necessity criteria.

    Call this from inside the agent loop when the mechanical criteria check is
    ``indeterminate`` / ``not_found`` or leaves unmet requirements. ``query`` is a
    natural-language description of what needs supporting; pass ``service_code``
    to scope the search to one policy.

    Returns a JSON-serializable list of citation dicts
    (``source, clause_id, quote, relevance``); an empty list is a valid
    "nothing relevant" answer.
    """
    settings = get_settings()
    try:
        hits = index.search(
            query, service_code=service_code, embedder=_get_tool_embedder()
        )
        if not hits or hits[0]["score"] < settings.rag_min_score:
            rewritten = query + _REWRITE_SUFFIX
            name = _service_name(service_code)
            if name:
                rewritten = f"{rewritten} {name}"
            hits = index.search(
                rewritten, service_code=service_code, embedder=_get_tool_embedder()
            )
            hits = [h for h in hits if h["score"] >= settings.rag_rewrite_min_score]
            hits = hits[: settings.rag_top_k]
    except RagIndexUnavailable:
        # Do not crash the agent: PR5's worker then goes indeterminate ->
        # human_review on an empty citation list.
        _log.warning("clinical-guidance index unavailable; returning no citations")
        return []

    return [
        CriteriaCitation(
            source="rag_corpus",
            clause_id=h["chunk_id"],
            quote=h["text"],
            relevance=f"{h['section']} · score {h['score']:.2f}",
        ).model_dump()
        for h in hits
    ]
