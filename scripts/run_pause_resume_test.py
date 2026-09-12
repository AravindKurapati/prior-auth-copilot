"""AC-05 pause/resume, two real processes.

    python scripts/run_pause_resume_test.py                 # writes traces/pause_resume_transcript.md
    python scripts/run_pause_resume_test.py --stdout-only    # print, don't write (manual determinism checks)

Process 1 ("pause"): builds the real compiled graph (`pa_copilot.graph.make_graph`,
fakes standing in for the model since there's no API key) against a real
`AsyncSqliteSaver` file, drives a scripted case straight to `human_review`'s
`interrupt()` on the very first supervisor turn, and exits -- the checkpointer
persists the paused state to the shared `.db` file.

Process 2 ("resume"): a brand-new interpreter, opens a FRESH `AsyncSqliteSaver`
connection (and a fresh `make_graph(...)` object) at the SAME db file, resumes
with `Command(resume=...)`, and asserts the case completes with no further
interrupt. Combined stdout (volatile bits masked: pids as the literal `<pid>`,
real timestamps as `<ts>`) is the committed evidence.

How graph1 is driven to `human_review` deterministically (verified empirically
against the installed `pa_copilot.supervisor.hard_route`, whose guardrail order
is hop-cap -> tool-failure-cap -> decision-finished -> request-None -> LLM):
the initial state seeds `supervisor_hops` at `settings.max_hops` (the hop-cap
guardrail wins immediately, before the "decision finished" check ever runs,
matching `test_hard_route_hop_cap_wins_over_finished_decision`) while ALSO
pre-populating a finished `decision` (with `needs_replan=False`). The resume
half's `Command(resume=..., update={"supervisor_hops": 0})` resets the hop
count as part of the resume -- `Command.update` is applied before the graph
re-enters the interrupted node, confirmed empirically this task -- so the
supervisor's *second* turn no longer trips the (otherwise permanently sticky:
hops only ever increase) hop-cap guardrail and instead falls through to the
"decision finished" rule, which routes straight to `FINISH`/`END`. Without the
`update=`, the hop-cap guardrail would fire again on every resumed turn and the
case would never complete -- verified empirically that plain
`Command(resume=...)` alone reproduces exactly that infinite-interrupt loop.

``sys.path[0]`` is ``scripts/`` when invoked this way, so put the repo root
(and ``src``) on the path first (Ruling R3, same as
``scripts/run_persistence_test.py``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

CASE_ID = "case-ac05-2proc"
THREAD = {"configurable": {"thread_id": CASE_ID}}
_TS = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+\d{2}:\d{2})?")


def _pid() -> str:
    return "<pid>"  # never a real pid -- byte-stable by construction


def _fake_embedder():
    """Bare-import the shared fakes (Task 6/R32 ruling, mirrors
    ``run_persistence_test.py``'s ``_fake()``): ``tests/`` is not a package, so
    this must be ``from _fakes import ...`` with ``tests/`` on ``sys.path`` --
    NOT ``from tests._fakes import``. Each subprocess is a fresh interpreter
    re-running this module, so the insert happens here (call time)."""
    sys.path.insert(0, str(_REPO_ROOT / "tests"))
    from _fakes import FakeEmbedder  # noqa: PLC0415

    return FakeEmbedder()


def _fake_tool_calling_model():
    sys.path.insert(0, str(_REPO_ROOT / "tests"))
    from _fakes import FakeToolCallingModel  # noqa: PLC0415

    return FakeToolCallingModel()


def _stub_out_summarizer() -> None:
    """`pa_copilot.context.summarization.summarize`'s module-level node cache
    builds a real `ChatGoogleGenerativeAI` on first use unless pre-seeded --
    verified empirically (this task) that constructing it with no Google/Gemini
    credentials configured raises `google.auth.exceptions.DefaultCredentialsError`
    immediately, at construction time, well before any network call or before it
    would ever actually be invoked (the scripted case's `messages` list is empty,
    so `SummarizationNode` short-circuits without invoking any model at all --
    but the module builds the model regardless, the first time `summarize()`
    runs). Each subprocess is a fresh interpreter, so this must be called once
    per process, same as the sys.path inserts above."""
    import pa_copilot.context.summarization as summarization_mod  # noqa: PLC0415
    from pa_copilot.context.summarization import build_summarization_node  # noqa: PLC0415

    summarization_mod._node = build_summarization_node(
        model=_fake_tool_calling_model(), max_tokens=100_000, max_tokens_before_summary=100_000
    )


async def _run_pause(state_db: str, mem_db: str) -> int:
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from pa_copilot.config import get_settings
    from pa_copilot.graph import make_graph
    from pa_copilot.memory.store import memory_store
    from pa_copilot.schemas import PADecision
    from pa_copilot.state import new_case_state

    _stub_out_summarizer()
    settings = get_settings()

    with memory_store(mem_db, embedder=_fake_embedder()) as store:
        conn = await aiosqlite.connect(state_db)
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        graph = await make_graph(
            store=store, checkpointer=saver, mcp_tools=[], model=_fake_tool_calling_model()
        )

        state = new_case_state(CASE_ID, "sess-ac05-2proc", "M100001", "raw provider text")
        state["supervisor_hops"] = settings.max_hops
        state["decision"] = PADecision(
            disposition="approve",
            reviewer_summary="auto-approved pending human sign-off",
            confidence=0.92,
            human_review_required=True,
        )
        state["needs_replan"] = False

        print(f"[pause pid={_pid()}] driving case {CASE_ID} to human_review's interrupt")
        result = await graph.ainvoke(state, config=THREAD)
        paused = "__interrupt__" in result
        if paused:
            payload = result["__interrupt__"][0].value
            print(
                f"[pause pid={_pid()}] interrupt payload: "
                f"{json.dumps(payload, sort_keys=True)}"
            )
        print(f"[pause pid={_pid()}] paused={paused}")
        await conn.close()
    return 0 if paused else 1


async def _run_resume(state_db: str, mem_db: str) -> int:
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import Command

    from pa_copilot.graph import make_graph
    from pa_copilot.memory.store import memory_store

    _stub_out_summarizer()

    with memory_store(mem_db, embedder=_fake_embedder()) as store:
        conn = await aiosqlite.connect(state_db)
        saver = AsyncSqliteSaver(conn)
        graph = await make_graph(
            store=store, checkpointer=saver, mcp_tools=[], model=_fake_tool_calling_model()
        )

        print(f"[resume pid={_pid()}] built a fresh graph object against the same state db")
        result = await graph.ainvoke(
            Command(resume="approved", update={"supervisor_hops": 0}), config=THREAD
        )
        completed = "__interrupt__" not in result
        decision = result.get("decision")
        print(
            f"[resume pid={_pid()}] completed={completed} next={result.get('next')} "
            f"disposition={decision.disposition if decision else None}"
        )
        await conn.close()
    return 0 if completed else 1


def _run_child(state_db: str, mem_db: str, mode: str) -> tuple[str, int]:
    res = subprocess.run(
        [sys.executable, __file__, "--state-db", state_db, "--mem-db", mem_db, "--_child", mode],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 and res.stderr:
        # Diagnostic-only: goes to the PARENT's own stderr so a human/CI log
        # shows the child's traceback. Never folded into `combined` (the
        # committed log) or the --stdout-only text -- both stay byte-stable.
        print(res.stderr, file=sys.stderr, end="")
    return _TS.sub("<ts>", res.stdout), res.returncode


def _cleanup(*paths: str) -> None:
    for path in paths:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(path + suffix).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-db", default=None, help="sqlite checkpoint path")
    ap.add_argument("--mem-db", default=None, help="sqlite memory-store path")
    ap.add_argument("--stdout-only", action="store_true")
    ap.add_argument("--_child", choices=["pause", "resume"], help=argparse.SUPPRESS)
    args = ap.parse_args()

    # Child invocation: this same file, run as a brand-new interpreter by
    # `_run_child` above. Just do the one thing and exit.
    if args._child == "pause":
        return asyncio.run(_run_pause(args.state_db, args.mem_db))
    if args._child == "resume":
        return asyncio.run(_run_resume(args.state_db, args.mem_db))

    # Parent/orchestrator invocation: spawn the two real child processes.
    owns_dbs = args.state_db is None
    if owns_dbs:
        fd1, state_db = tempfile.mkstemp(suffix=".db", prefix="pause_resume_state_")
        os.close(fd1)
        fd2, mem_db = tempfile.mkstemp(suffix=".db", prefix="pause_resume_mem_")
        os.close(fd2)
    else:
        state_db, mem_db = args.state_db, args.mem_db
    _cleanup(state_db, mem_db)  # start from a clean slate either way

    try:
        pause_out, pause_rc = _run_child(state_db, mem_db, "pause")
        resume_out, resume_rc = _run_child(state_db, mem_db, "resume")
    finally:
        _cleanup(state_db, mem_db)

    passed = pause_rc == 0 and resume_rc == 0
    combined = (
        "# AC-05 pause/resume -- two python processes, one shared checkpoint db\n"
        f"{pause_out}{resume_out}"
        f"RESULT: {'PASS' if passed else 'FAIL'}\n"
    )

    if args.stdout_only:
        sys.stdout.write(combined)
    else:
        (_REPO_ROOT / "traces" / "pause_resume_transcript.md").write_text(
            combined, encoding="utf-8", newline="\n"
        )
        print("wrote traces/pause_resume_transcript.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
