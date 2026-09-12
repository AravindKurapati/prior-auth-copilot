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
from langmem.short_term import SummarizationNode

from pa_copilot.agents._react import get_agent_model
from pa_copilot.config import get_settings
from pa_copilot.state import PACaseState

# Module-level cache for the node `summarize()` builds lazily on first use --
# mirrors `rag/tool.py`'s `_default_embedder` pattern (build once, reuse; the
# real default model is a network-backed client, not something to reconstruct
# per call).
_node: SummarizationNode | None = None


def _default_model() -> BaseChatModel:
    """`settings.model_summarizer` is deliberately the cheaper model
    (`gemini-flash-lite-latest` per design.md §1), distinct from the agent
    model workers use -- so this passes `model_name` explicitly rather than
    taking `get_agent_model()`'s `.model_agent` default. Constructor logic
    itself lives in `agents/_react.py::get_agent_model()` (Task 6); this is a
    thin re-import, not a duplicate, per that task's ordering note."""
    s = get_settings()
    return get_agent_model(model_name=s.model_summarizer, settings=s)


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
