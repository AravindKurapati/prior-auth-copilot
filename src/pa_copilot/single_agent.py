"""The single-agent baseline (design.md §10 PR7, NFR-06's comparison requirement):
one ReAct loop bound to every tool the four multi-agent workers collectively use,
no supervisor, no per-worker context selection, no quarantine boundary beyond what
the one system prompt states in words. Exists so `pac compare` has a REAL point of
comparison, not a strawman.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from pa_copilot.agents._react import get_agent_model, get_lite_agent_model
from pa_copilot.memory.store import PolicyStore
from pa_copilot.memory.tools import build_memory_tools
from pa_copilot.rag.tool import search_clinical_guidance
from pa_copilot.reflection import attribute_tool_errors, run_worker_react_resilient
from pa_copilot.schemas import PADecision

_SYSTEM_PROMPT = (
    "You are a single agent responsible for the ENTIRE prior-authorization "
    "adjudication task: extract the structured request from the raw provider "
    "submission, check plan benefits, assess medical necessity against payer "
    "criteria (using tools as needed), and produce a final PADecision. You have "
    "no supervisor and no specialized sub-agents -- do all of this yourself, "
    "calling tools directly."
)


async def run_single_agent(
    raw_provider_text: str,
    member_id: str,
    *,
    mcp_tools: list[BaseTool],
    store: PolicyStore,
    model=None,
) -> PADecision:
    manage, search_member = build_memory_tools(store, ("pa", "member", member_id))
    tools = attribute_tool_errors(
        [*mcp_tools, search_clinical_guidance, manage, search_member]
    )
    _messages, decision = await run_worker_react_resilient(
        model or get_agent_model(),
        tools,
        system_prompt=_SYSTEM_PROMPT,
        messages=[("user", raw_provider_text)],
        response_format=PADecision,
        # Deferred like every PR6 worker: only build a real lite fallback when
        # the caller didn't already supply a substitute `model` -- constructing
        # get_lite_agent_model() unconditionally would eagerly hit real Google
        # credential resolution even in fake-model tests (see medical_necessity.py).
        lite_model=get_lite_agent_model() if model is None else None,
    )
    return decision
