"""The hand-rolled StateGraph (design.md §3.1): assembles every worker,
supervisor, and summarize node into the single agentic loop. Pure assembly —
every component wired here already exists and is independently tested
elsewhere; this module only owns the topology.

    START -> summarize -> supervisor
    supervisor --conditional on state["next"]--> {intake|benefit_check
        |medical_necessity|decision_draft|human_review|END}
    intake, benefit_check, medical_necessity, decision_draft, human_review
        -> summarize -> supervisor

mcp_tools filtering: passed as the full list to every worker that takes tools
(the simplest option the task brief asked to try first) rather than a
per-worker subset — each worker's own system prompt already scopes which tool
it calls, so an unused tool being technically bindable is harmless; nothing in
this PR's tests showed a worker mis-calling an irrelevant tool, so no split
was needed. decision_draft's factory takes no mcp_tools parameter at all (it
never calls tools), so it is not passed any.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from pa_copilot.agents.benefit_check import build_benefit_check_node
from pa_copilot.agents.decision_draft import build_decision_draft_node
from pa_copilot.agents.human_review import build_human_review_node
from pa_copilot.agents.intake import build_intake_node
from pa_copilot.agents.medical_necessity import build_medical_necessity_node
from pa_copilot.context.summarization import summarize
from pa_copilot.memory.store import PolicyStore
from pa_copilot.state import PACaseState
from pa_copilot.supervisor import build_supervisor_node


async def make_graph(*, store: PolicyStore, checkpointer, mcp_tools: list, model=None):
    graph = StateGraph(PACaseState)

    graph.add_node("summarize", summarize)
    graph.add_node("supervisor", build_supervisor_node(model=model))
    graph.add_node("intake", build_intake_node(store=store, mcp_tools=mcp_tools, model=model))
    graph.add_node("benefit_check", build_benefit_check_node(mcp_tools=mcp_tools, model=model))
    graph.add_node(
        "medical_necessity",
        build_medical_necessity_node(mcp_tools=mcp_tools, model=model),
    )
    graph.add_node("decision_draft", build_decision_draft_node(store=store, model=model))
    graph.add_node("human_review", build_human_review_node())

    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {
            "intake": "intake",
            "benefit_check": "benefit_check",
            "medical_necessity": "medical_necessity",
            "decision_draft": "decision_draft",
            "human_review": "human_review",
            "FINISH": END,
        },
    )
    for worker in (
        "intake", "benefit_check", "medical_necessity", "decision_draft", "human_review",
    ):
        graph.add_edge(worker, "summarize")

    return graph.compile(checkpointer=checkpointer, store=store)
