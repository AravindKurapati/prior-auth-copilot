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

from pa_copilot.config import get_settings
from pa_copilot.rag import index
from pa_copilot.rag.embedder import Embedder
from pa_copilot.rag.index import RagIndexUnavailable
from pa_copilot.schemas import CriteriaCitation

_log = logging.getLogger(__name__)

_REWRITE_SUFFIX = " medical necessity criteria indications"

# The module-level default embedder — built once, lazily, on first use and then
# reused for every tool call (the real ``BgeEmbedder`` lazy-loads a ~130MB
# sentence-transformers model per instance, so a fresh instance per call would
# reload it, twice on the corrective-rewrite path).
_default_embedder: Embedder | None = None
# ``set_tool_embedder`` slot — tests point this at a ``FakeEmbedder`` so no model
# is downloaded; ``reset_tool_embedder`` clears it back to ``None`` without
# discarding the cached real default.
_tool_embedder_override: Embedder | None = None

# ``service_code -> policy title`` lookup, built lazily from the MCP data-access
# layer (``criteria.json``) on first use. Imported here rather than from
# ``data.synthetic.generators`` so this module is importable outside the repo root
# (``data/`` is not a shipped package; PR5's worker + PR7's console script run
# from other cwds).
_service_title_by_code: dict[str, str] | None = None


def set_tool_embedder(embedder: Embedder) -> None:
    """Force ``search_clinical_guidance`` to use ``embedder`` (test seam)."""
    global _tool_embedder_override
    _tool_embedder_override = embedder


def reset_tool_embedder() -> None:
    """Drop any override set by :func:`set_tool_embedder`.

    Leaves the cached real default in place — it is safe to keep.
    """
    global _tool_embedder_override
    _tool_embedder_override = None


def _get_tool_embedder() -> Embedder:
    global _default_embedder
    if _tool_embedder_override is not None:
        return _tool_embedder_override
    if _default_embedder is None:
        # Lazy: never construct the sentence-transformers model at import time.
        from pa_copilot.rag.embedder import BgeEmbedder

        settings = get_settings()
        _default_embedder = BgeEmbedder(
            settings.embedding_model, query_prefix=settings.rag_query_prefix
        )
    return _default_embedder


def _service_name(service_code: str | None) -> str | None:
    if not service_code:
        return None
    global _service_title_by_code
    if _service_title_by_code is None:
        from pa_copilot.mcp_server.data_access import list_policies

        _service_title_by_code = {
            p["service_code"]: p["title"] for p in list_policies()["policies"]
        }
    return _service_title_by_code.get(service_code)


def should_search_guidance(
    criteria_status: str,
    *,
    unmet_requirements: list[str] | None = None,
) -> bool:
    """Predicate PR5's ``medical_necessity`` worker calls before retrieval.

    Retrieval stays *inside* the agent loop rather than being a fixed pipeline
    step: the worker only reaches for the narrative guidance when the mechanical
    MCP ``criteria_check`` cannot settle the question.

    ``criteria_status`` takes two overlapping vocabularies:

    - the MCP ``criteria_check`` status (``not_found`` / ``excluded`` /
      ``indeterminate``) on the pre-assessment path — there this reduces to
      ``status != "excluded"``;
    - the worker's own ``NecessityAssessment.criteria_status`` (``met`` /
      ``not_met`` / ``indeterminate``) on a re-assessment loop.

    Returns ``True`` when:

    - ``criteria_status == "indeterminate"`` — policy exists but only a human /
      the narrative can verify the conditions;
    - ``criteria_status == "not_found"`` — no structured policy at all, the
      narrative is all we have;
    - ``criteria_status == "not_met"`` — a mechanical/clinical fail where the
      narrative interpretation matters most (a borderline gap may still be met
      against the guidance);
    - ``unmet_requirements`` is non-empty — checklist gaps that need
      interpretation against the guidance.

    Returns ``False`` for ``"excluded"`` (a mechanical deny signal — no narrative
    needed) and for a clear ``"met"``.
    """
    if criteria_status == "excluded":
        return False
    if criteria_status in ("indeterminate", "not_found", "not_met"):
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
    (``source, clause_id, quote, relevance``), every one scoring at or above
    ``rag_min_score``. An empty list is a valid "nothing relevant" answer and is
    also what a *still-weak* retrieval returns after the one corrective rewrite —
    PR5's worker reads empty citations as "still weak" → ``indeterminate`` →
    ``human_review`` (uniform with the index-unavailable path).
    """
    settings = get_settings()
    min_score = settings.rag_min_score
    try:
        hits = index.search(
            query, service_code=service_code, embedder=_get_tool_embedder()
        )
        kept = [h for h in hits if h["score"] >= min_score]
        if not kept:
            # One corrective rewrite. Keep whichever of {original, rewritten}
            # ranks better at the top, then hold *that* set to the same
            # ``rag_min_score`` bar the originals just failed — the rewrite path
            # must not be more permissive than the first search (I2).
            rewritten = query + _REWRITE_SUFFIX
            name = _service_name(service_code)
            if name:
                rewritten = f"{rewritten} {name}"
            rw_hits = index.search(
                rewritten, service_code=service_code, embedder=_get_tool_embedder()
            )
            orig_top = hits[0]["score"] if hits else 0.0
            rw_top = rw_hits[0]["score"] if rw_hits else 0.0
            best = rw_hits if rw_top >= orig_top else hits
            kept = [h for h in best if h["score"] >= min_score]
            if not kept:
                # Nothing clears the bar even after the rewrite → "still weak".
                return []
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
        for h in kept
    ]
