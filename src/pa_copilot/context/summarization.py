"""NFR-08: context-window compression. Wraps langmem's SummarizationNode, whose
contract (verified against the installed source,
``langmem/short_term/summarization.py::SummarizationNode._prepare_state_update``)
is: read ``state["context"]``, write ``{"summarized_messages": [...], "context":
{"running_summary": ...}}`` back -- only the second key is present once a summary
has actually fired. ``context`` is exactly the ``PACaseState`` field design.md
reserved for this.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langmem.short_term import SummarizationNode

from pa_copilot.config import get_settings
from pa_copilot.state import PACaseState

# Module-level cache for the node `summarize()` builds lazily on first use --
# mirrors `rag/tool.py`'s `_default_embedder` pattern (build once, reuse; the
# real default model is a network-backed client, not something to reconstruct
# per call).
_node: SummarizationNode | None = None


def _default_model() -> BaseChatModel:
    """Ordering wrinkle (task brief, PR5a plan): this tiny model-constructor
    piece is implemented here, ahead of Task 6's `agents/_react.py`, because
    NFR-08's evidence doesn't depend on the ReAct loop at all -- only on this
    two-line function. A later task moves this logic into
    `agents/_react.py::get_agent_model()` and this module re-imports it from
    there instead of duplicating it. `settings.model_summarizer` is
    deliberately the cheaper model (`gemini-flash-lite-latest` per design.md
    §1), distinct from the agent model workers use."""
    s = get_settings()
    return ChatGoogleGenerativeAI(model=s.model_summarizer, temperature=s.temperature_agent)


def build_summarization_node(
    model: BaseChatModel | None = None,
    *,
    max_tokens: int = 2048,
    max_tokens_before_summary: int = 1200,
    max_summary_tokens: int = 256,
) -> SummarizationNode:
    return SummarizationNode(
        model=model or _default_model(),
        max_tokens=max_tokens,
        max_tokens_before_summary=max_tokens_before_summary,
        max_summary_tokens=max_summary_tokens,
    )


async def summarize(state: PACaseState) -> dict:
    """Graph-node-shaped wrapper. PR5b's `graph.py` wires this directly as the
    `summarize` node (design.md §3.1 topology: runs before every supervisor
    turn). Returns whatever dict the underlying node produces -- either
    `{"summarized_messages": [...], "context": {...}}` when a summary fired, or
    `{"summarized_messages": [...]}` alone when the thread is still under
    threshold."""
    global _node
    if _node is None:
        _node = build_summarization_node()
    return await _node.ainvoke(state)
