"""AC-05: a checkpointer persists graph state so a case can be paused and
resumed. In-process teardown+rebuild form (genuine 2-process form is
scripts/run_pause_resume_test.py -> traces/pause_resume_transcript.md).

Driving graph1 to `human_review` deterministically: verified against the
installed `pa_copilot.supervisor.hard_route` (guardrail order: hop-cap ->
tool-failure-cap -> decision-finished -> request-None -> LLM). Seeding
`state["supervisor_hops"]` at `settings.max_hops` makes the hop-cap guardrail
win on the very first supervisor turn (same guardrail
`test_hard_route_hop_cap_wins_over_finished_decision` pins), skipping the need
to script any worker's LLM call. The hop-cap guardrail is otherwise permanently
sticky (hops only ever increase, so it would keep firing forever on later
turns too) -- so a finished `decision` (`needs_replan=False`) is ALSO seeded
up front, and the resume half passes `Command(resume=..., update=
{"supervisor_hops": 0})`: `Command.update` is applied before the graph
re-enters the interrupted node (confirmed empirically this task), so the
supervisor's second turn no longer trips the hop cap and instead falls through
to the "decision finished" rule, routing straight to FINISH/END. Before PR6,
plain `Command(resume=...)` alone (no `update=`) reproduced an infinite
interrupt loop instead -- the hop-cap guardrail simply fired again on every
resumed turn. PR6 Task 4 closed that gap at the source: `human_review.py`'s
node now resets `supervisor_hops`/`replan_count` to 0 itself on resume, so a
manual `update=` is no longer required (see
`test_resume_completes_without_a_manual_hop_reset` below) -- the explicit
`update={"supervisor_hops": 0}` this test still passes is now redundant with
that automatic reset, not load-bearing, and kept only because both agree.
"""

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
import aiosqlite

from pa_copilot.config import get_settings
from pa_copilot.graph import make_graph
from pa_copilot.memory.store import memory_store
from pa_copilot.schemas import PADecision
from pa_copilot.state import new_case_state
from _fakes import FakeToolCallingModel


@pytest.mark.asyncio
async def test_interrupt_then_resume_from_a_fresh_graph_object(
    tmp_path, fake_embedder, monkeypatch
):
    # `summarize()`'s module-level node cache builds a real ChatGoogleGenerativeAI
    # on first use unless pre-seeded -- verified empirically that this raises
    # DefaultCredentialsError at construction time with no Google/Gemini
    # credentials configured, mirrors
    # test_context_summarization.py::test_summarize_node_wrapper_is_graph_node_shaped.
    import pa_copilot.context.summarization as summarization_mod
    from pa_copilot.context.summarization import build_summarization_node

    monkeypatch.setattr(
        summarization_mod,
        "_node",
        build_summarization_node(
            model=FakeToolCallingModel(script=[]),
            max_tokens=100_000,
            max_tokens_before_summary=100_000,
        ),
    )

    db_path = str(tmp_path / "state.db")
    thread = {"configurable": {"thread_id": "case-ac05"}}
    settings = get_settings()

    with memory_store(tmp_path / "mem.db", embedder=fake_embedder) as store:
        conn1 = await aiosqlite.connect(db_path)
        saver1 = AsyncSqliteSaver(conn1)
        await saver1.setup()
        graph1 = await make_graph(
            store=store, checkpointer=saver1, mcp_tools=[], model=FakeToolCallingModel()
        )

        initial_state = new_case_state("case-ac05", "sess-ac05", "M100001", "raw provider text")
        initial_state["supervisor_hops"] = settings.max_hops
        initial_state["decision"] = PADecision(
            disposition="approve",
            reviewer_summary="auto-approved pending human sign-off",
            confidence=0.92,
            human_review_required=True,
        )
        initial_state["needs_replan"] = False

        result1 = await graph1.ainvoke(initial_state, config=thread)
        assert "__interrupt__" in result1
        interrupt_payload = result1["__interrupt__"][0].value
        assert interrupt_payload["reason"] == "human review required"
        await conn1.close()

        # A FRESH graph/connection object -- new AsyncSqliteSaver over a new
        # aiosqlite connection to the SAME file, new make_graph(...) call.
        # This is the whole point of the proof: nothing from graph1's Python
        # objects is reused, only the on-disk state.
        conn2 = await aiosqlite.connect(db_path)
        saver2 = AsyncSqliteSaver(conn2)
        graph2 = await make_graph(
            store=store, checkpointer=saver2, mcp_tools=[], model=FakeToolCallingModel()
        )
        result = await graph2.ainvoke(
            Command(resume="approved", update={"supervisor_hops": 0}), config=thread
        )
        assert "__interrupt__" not in result
        assert result["next"] == "FINISH"
        assert result["decision"].disposition == "approve"
        await conn2.close()


@pytest.mark.asyncio
async def test_resume_completes_without_a_manual_hop_reset(tmp_path, fake_embedder, monkeypatch):
    """PR6 Task 4: `human_review.py`'s node now resets `supervisor_hops`/
    `replan_count` to 0 itself on resume (a human just intervened -- fresh
    attempt budget), closing the gap this test used to pin (formerly
    `test_resuming_without_the_hop_reset_reinterrupts`, which asserted a bare
    `Command(resume=...)` with no manual `update=` re-triggered the sticky
    hop-cap guardrail forever). The other test in this file
    (`test_interrupt_then_resume_from_a_fresh_graph_object`) still passes
    `update={"supervisor_hops": 0}` explicitly -- now redundant with the
    automatic reset (both set it to 0), not wrong, so left as-is. This test
    proves the reset also works with NO manual update at all, i.e. the fix
    lives in the node, not in every caller."""
    import pa_copilot.context.summarization as summarization_mod
    from pa_copilot.context.summarization import build_summarization_node

    monkeypatch.setattr(
        summarization_mod,
        "_node",
        build_summarization_node(
            model=FakeToolCallingModel(script=[]),
            max_tokens=100_000,
            max_tokens_before_summary=100_000,
        ),
    )

    db_path = str(tmp_path / "state.db")
    thread = {"configurable": {"thread_id": "case-ac05-no-reset"}}
    settings = get_settings()

    with memory_store(tmp_path / "mem.db", embedder=fake_embedder) as store:
        conn1 = await aiosqlite.connect(db_path)
        saver1 = AsyncSqliteSaver(conn1)
        await saver1.setup()
        graph1 = await make_graph(
            store=store, checkpointer=saver1, mcp_tools=[], model=FakeToolCallingModel()
        )

        initial_state = new_case_state("c2", "s2", "M100001", "raw text")
        initial_state["supervisor_hops"] = settings.max_hops
        initial_state["decision"] = PADecision(
            disposition="approve", reviewer_summary="x", confidence=0.9,
            human_review_required=True,
        )
        initial_state["needs_replan"] = False

        result1 = await graph1.ainvoke(initial_state, config=thread)
        assert "__interrupt__" in result1
        await conn1.close()

        conn2 = await aiosqlite.connect(db_path)
        saver2 = AsyncSqliteSaver(conn2)
        graph2 = await make_graph(
            store=store, checkpointer=saver2, mcp_tools=[], model=FakeToolCallingModel()
        )
        result = await graph2.ainvoke(Command(resume="approved"), config=thread)
        assert "__interrupt__" not in result  # completes: human_review's own reset
        assert result["next"] == "FINISH"
        # human_review resets supervisor_hops to 0, but the graph doesn't stop
        # there -- it continues through summarize -> supervisor, and
        # build_supervisor_node increments supervisor_hops by 1 on EVERY turn
        # (that's how the hop cap counts turns at all). So the final state
        # after one post-resume supervisor turn is 0 + 1 = 1, not 0 -- this is
        # the reset actually working (without it, that one turn would have
        # incremented from max_hops to max_hops + 1, immediately re-tripping
        # the cap instead of reaching "decision finished" -> FINISH).
        assert result["supervisor_hops"] == 1
        assert result["replan_count"] == 0
        await conn2.close()
