"""`pac` — the CLI surface over the already-tested pa_copilot pipeline (design.md
§8.1, PR7). Every subcommand wires real dependencies (store, checkpointer, MCP
client) and calls into existing, independently-tested modules — no new business
logic lives here.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
import typer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from pa_copilot.config import Settings, get_settings
from pa_copilot.graph import make_graph
from pa_copilot.mcp_client import build_client, load_pa_tools, pa_session
from pa_copilot.memory.store import PolicyStore, memory_store, open_memory_store
from pa_copilot.rag import corpus
from pa_copilot.rag.index import build_index
from pa_copilot.schemas import SampleSubmission
from pa_copilot.state import new_case_state
from pa_copilot.tracing import RunTracer, default_redact_values

app = typer.Typer(help="Prior-authorization copilot CLI.")
memory_app = typer.Typer(help="Inspect long-term memory.")
app.add_typer(memory_app, name="memory")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_namespace(namespace: str) -> tuple[str, ...]:
    return tuple(namespace.split(","))


def _real_embedder(settings: Settings):
    """The same embedder construction rag/index.py's `_resolve` uses -- lazy
    (no model load until first embed call), so importing/constructing this is
    cheap even when a command never ends up calling it."""
    from pa_copilot.rag.embedder import BgeEmbedder  # noqa: PLC0415

    return BgeEmbedder(settings.embedding_model)


@asynccontextmanager
async def _open_state(settings: Settings):
    """One aiosqlite connection + one PolicyStore, held open for the caller's
    whole graph invocation (design.md §3.7 -- both must stay open for the
    graph's lifetime, never closed via `from_conn_string`'s `__aexit__`)."""
    conn = await aiosqlite.connect(settings.state_db)
    try:
        checkpointer = AsyncSqliteSaver(conn)
        with memory_store(settings.memory_db, settings=settings, embedder=_real_embedder(settings)) as store:
            yield checkpointer, store
    finally:
        await conn.close()


@asynccontextmanager
async def _open_mcp_tools():
    """One live stdio MCP session for the caller's whole use of the returned
    tools -- the session (and its subprocess) stays open until the `async with`
    block exits, so every tool call inside it reuses the same live process
    instead of one that already closed."""
    client = build_client()
    async with pa_session(client) as session:
        yield await load_pa_tools(session=session)


@asynccontextmanager
async def _open_graph(store: PolicyStore, checkpointer):
    """Real MCP tools over one live stdio session, wired into the real compiled
    graph, kept open for the whole `async with` block -- the graph must not be
    used after this block exits (its tools' underlying session would already be
    closed). Kept as one seam so tests can monkeypatch it wholesale with a fake
    model + FAKE_MCP_TOOLS (see tests/_full_case.py) without touching the rest
    of a command's logic."""
    async with _open_mcp_tools() as mcp_tools:
        yield await make_graph(store=store, checkpointer=checkpointer, mcp_tools=mcp_tools)


@app.command()
def ingest() -> None:
    """Build the clinical-guidance RAG index and verify synthetic MCP data is present."""
    settings = get_settings()
    synthetic_dir = Path(settings.synthetic_dir)
    required = ["benefits.json", "criteria.json", "providers.json"]
    missing = [f for f in required if not (synthetic_dir / f).exists()]
    if missing:
        typer.echo(
            f"Missing synthetic data files under {synthetic_dir}: {missing}. "
            "Run the PR1 generator scripts first.",
            err=True,
        )
        raise typer.Exit(code=1)

    chunks_path = Path(settings.data_dir) / "synthetic" / "clinical_guidance_chunks.jsonl"
    corpus.write_chunks_jsonl(
        chunks_path, corpus.load_guidance(synthetic_dir / "clinical_guidance")
    )
    typer.echo(f"wrote {chunks_path}")

    summary = build_index(settings)
    typer.echo(f"RAG index built: {dataclasses.asdict(summary)}")


@memory_app.command("list")
def memory_list(namespace: str) -> None:
    """List keys under a memory namespace, e.g. 'pa,member,M100001'."""
    ns = _parse_namespace(namespace)
    settings = get_settings()
    store = open_memory_store(settings.memory_db, settings=settings, semantic=False)
    try:
        items = asyncio.run(store.asearch(ns))
        for item in items:
            typer.echo(f"{item.key}: {item.value}")
    finally:
        store.conn.close()


@memory_app.command("search")
def memory_search(namespace: str, query: str, limit: int = 5) -> None:
    """Semantic/keyword search a memory namespace."""
    ns = _parse_namespace(namespace)
    settings = get_settings()
    store = open_memory_store(
        settings.memory_db, settings=settings, semantic=True, embedder=_real_embedder(settings)
    )
    try:
        items = asyncio.run(store.asearch(ns, query=query, limit=limit))
        for item in items:
            score = f"{item.score:.3f}" if item.score is not None else "n/a"
            typer.echo(f"{score}  {item.key}: {item.value}")
    finally:
        store.conn.close()


@memory_app.command("show")
def memory_show(namespace: str, key: str) -> None:
    """Show one memory item by exact key."""
    ns = _parse_namespace(namespace)
    settings = get_settings()
    store = open_memory_store(settings.memory_db, settings=settings, semantic=False)
    try:
        item = asyncio.run(store.aget(ns, key))
        if item is None:
            typer.echo(f"no item at {ns}/{key}", err=True)
            raise typer.Exit(code=1)
        typer.echo(item.value)
    finally:
        store.conn.close()


@app.command("persistence-test")
def persistence_test() -> None:
    """Run the cross-session persistence evidence script -> traces/memory_persistence.log."""
    script = _REPO_ROOT / "scripts" / "run_persistence_test.py"
    subprocess.run([sys.executable, str(script)], check=True)


@app.command()
def submit(
    sample_path: Path,
    session_id: str = typer.Option(None, "--session-id"),
    member_id: str = typer.Option(None, "--member-id"),
) -> None:
    """Run one request through the graph; print the routing trail + decision; write a trace."""
    settings = get_settings()
    raw = json.loads(sample_path.read_text(encoding="utf-8"))
    submission = SampleSubmission(**raw)
    case_id = submission.case_id
    sid = session_id or submission.session_id
    mid = member_id or submission.member_id

    async def _run():
        async with _open_state(settings) as (checkpointer, store):
            thread = {"configurable": {"thread_id": case_id}}
            existing = await checkpointer.aget_tuple(thread)
            if existing is not None:
                typer.echo(
                    f"case_id={case_id} already has a checkpoint (from a prior "
                    f"submit/resume) -- use `pac resume {case_id}` instead of "
                    "re-submitting, to avoid running fresh state against an "
                    "already-in-progress thread.",
                    err=True,
                )
                raise typer.Exit(code=1)

            tracer = RunTracer(
                settings.traces_dir,
                case_id,
                redact_values=default_redact_values(Path(settings.data_dir)),
                session_id=sid,
            )
            result = None
            try:
                async with _open_graph(store, checkpointer) as graph:
                    state = new_case_state(case_id, sid, mid, submission.raw_provider_text)
                    result = await graph.ainvoke(state, config=thread)
            finally:
                for step in (result or {}).get("route_history") or []:
                    tracer.event(step.to_node, "route", {"reason": step.reason})
                decision = (result or {}).get("decision")
                tracer.finish(decision=decision.model_dump() if decision else None)

            typer.echo(f"case_id={case_id}")
            if decision:
                typer.echo(json.dumps(decision.model_dump(), indent=2))
            elif "__interrupt__" in (result or {}):
                typer.echo(f"paused -- resume with: pac resume {case_id}")
            else:
                typer.echo("no decision reached (see trace for detail)")

    asyncio.run(_run())


@app.command()
def resume(case_id: str, value: str = typer.Argument("approved")) -> None:
    """Resume a paused case from the checkpointer -- a separate process invocation.

    Relies entirely on human_review.py's existing resume-time reset of
    supervisor_hops/replan_count (PR6) -- no manual Command(update=...) override
    belongs here.
    """
    settings = get_settings()

    async def _run():
        async with _open_state(settings) as (checkpointer, store):
            thread = {"configurable": {"thread_id": case_id}}
            tracer = RunTracer(
                settings.traces_dir,
                case_id,
                redact_values=default_redact_values(Path(settings.data_dir)),
            )
            result = None
            try:
                async with _open_graph(store, checkpointer) as graph:
                    result = await graph.ainvoke(Command(resume=value), config=thread)
            finally:
                for step in (result or {}).get("route_history") or []:
                    tracer.event(step.to_node, "route", {"reason": step.reason})
                decision = (result or {}).get("decision")
                tracer.finish(decision=decision.model_dump() if decision else None)

            if decision:
                typer.echo(json.dumps(decision.model_dump(), indent=2))
            elif "__interrupt__" in (result or {}):
                typer.echo(f"paused again -- resume with: pac resume {case_id}")
            else:
                typer.echo("no decision -- case may still be incomplete")

    asyncio.run(_run())


@app.command()
def demo() -> None:
    """Launch the Streamlit UI."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(_REPO_ROOT / "app" / "streamlit_app.py"),
        ],
        check=True,
    )


@app.command(name="all")
def run_all() -> None:
    """ingest -> sample battery -> persistence test -> compare (single documented
    command, NFR-02)."""
    ingest()
    settings = get_settings()
    samples = sorted(Path(settings.samples_dir).glob("*.json"))
    failed: set[Path] = set()
    for sample in samples:
        typer.echo(f"--- submitting {sample.name} ---")
        try:
            submit(sample, session_id=None, member_id=None)
        except Exception as exc:  # noqa: BLE001 -- one bad sample must not abort the battery
            typer.echo(f"  FAILED: {exc}", err=True)
            failed.add(sample)
    persistence_test()
    remaining = [s for s in samples if s not in failed]
    if remaining:
        try:
            compare(remaining[0])
        except Exception as exc:  # noqa: BLE001 -- compare is illustrative, not a hard gate
            typer.echo(f"compare FAILED: {exc}", err=True)


@app.command()
def compare(sample_path: Path) -> None:
    """Run one sample through both the multi-agent graph and the single-agent
    baseline; print both decisions + an observation."""
    from pa_copilot.single_agent import run_single_agent  # noqa: PLC0415

    settings = get_settings()
    raw = json.loads(sample_path.read_text(encoding="utf-8"))
    submission = SampleSubmission(**raw)

    async def _run():
        async with _open_state(settings) as (checkpointer, store):
            case_id = f"{submission.case_id}-multi"
            thread = {"configurable": {"thread_id": case_id}}
            async with _open_mcp_tools() as mcp_tools:
                graph = await make_graph(store=store, checkpointer=checkpointer, mcp_tools=mcp_tools)
                state = new_case_state(
                    case_id, submission.session_id, submission.member_id,
                    submission.raw_provider_text,
                )
                multi_result = await graph.ainvoke(state, config=thread)
                multi_decision = multi_result.get("decision")

                single_decision = await run_single_agent(
                    submission.raw_provider_text, submission.member_id,
                    mcp_tools=mcp_tools, store=store,
                )

            typer.echo("multi-agent decision:")
            if multi_decision:
                typer.echo(json.dumps(multi_decision.model_dump(), indent=2))
            elif "__interrupt__" in multi_result:
                typer.echo(f"paused -- resume with: pac resume {case_id}")
            else:
                typer.echo("null (no decision reached)")
            typer.echo("single-agent decision:")
            typer.echo(json.dumps(single_decision.model_dump(), indent=2))

    asyncio.run(_run())


if __name__ == "__main__":
    app()
