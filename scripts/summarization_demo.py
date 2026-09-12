"""Deterministic producer of traces/context_before_after.md (NFR-08). Builds a
long synthetic provider-note thread, runs it through
context.summarization.build_summarization_node with a scripted FakeToolCallingModel
(no network call -- this trace is reproducible without a Gemini key, unlike the
MCP/RAG traces which need the real thing where noted), and writes a before/after
token-count comparison.

    python scripts/summarization_demo.py

``sys.path[0]`` is ``scripts/`` when invoked this way, so put the repo root (and
``src``) on the path first (mirrors ``scripts/run_persistence_test.py`` /
``scripts/memory_demo.py``).

Note: ``count_tokens_approximately`` lives at ``langchain_core.messages.utils``,
not re-exported from ``langmem.short_term`` (verified empirically against the
installed langmem 0.0.x -- ``langmem.short_term.__all__`` does not include it;
``langmem``'s own ``summarization.py`` imports it from ``langchain_core``
internally, which is what this module mirrors).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.messages.utils import count_tokens_approximately  # noqa: E402

from pa_copilot.context.summarization import build_summarization_node  # noqa: E402

_SUMMARY_TEXT = (
    "Summary: member has completed 8+ weeks of conservative physical therapy for "
    "lumbar radiculopathy with persistent symptoms; multiple provider check-ins "
    "documented ongoing pain and no significant improvement."
)


def _long_thread(n: int = 24) -> list:
    msgs = []
    for i in range(n):
        msgs.append(
            HumanMessage(
                id=f"h{i}",
                content=(
                    f"Provider check-in {i}: patient continues physical therapy for "
                    "lumbar radiculopathy, reports persistent pain, no improvement "
                    "noted this session, plan to continue conservative treatment."
                ),
            )
        )
        msgs.append(AIMessage(id=f"a{i}", content=f"Acknowledged check-in {i}."))
    return msgs


def _fake_model():
    """Bare-import the shared fake test double (mirrors ``run_persistence_test.py``'s
    ``_fake()`` / ``memory_demo.py``'s ``_fake()``): ``tests/`` is not a package,
    so this is ``from _fakes import FakeToolCallingModel`` with ``tests/`` put on
    ``sys.path`` at call time -- not ``from tests._fakes import ...``. Deliberate
    demo-only use of a test double as evidence infrastructure: no network call,
    so this trace is reproducible without a Gemini key."""
    sys.path.insert(0, str(_REPO_ROOT / "tests"))
    from _fakes import FakeToolCallingModel  # noqa: PLC0415

    return FakeToolCallingModel(script=[AIMessage(content=_SUMMARY_TEXT)])


async def run_demo(*, trace_dir: Path | str = _REPO_ROOT / "traces") -> dict:
    """Async so tests can `await` it directly under pytest-asyncio's own event
    loop rather than nesting a second `asyncio.run()` inside a running pytest
    session (empirically: doing that intermittently left an unclosed loopback
    socket to be GC'd during an unrelated later test on this Windows box --
    likely event-loop-policy interaction between pytest-asyncio's per-test loop
    and a bare `asyncio.run()` call. Awaiting directly avoids creating that
    second loop). The `__main__` entrypoint below is the one legitimate place
    to call `asyncio.run()`, since nothing else owns a loop there."""
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    messages = _long_thread()
    tokens_before = count_tokens_approximately(messages)

    # max_summary_tokens must be < max_tokens (library invariant) -- pinned well
    # below both max_tokens=300 and max_tokens_before_summary=150.
    node = build_summarization_node(
        model=_fake_model(), max_tokens=300, max_tokens_before_summary=150, max_summary_tokens=50
    )
    result = await node.ainvoke({"messages": messages, "context": {}})

    tokens_after = count_tokens_approximately(result["summarized_messages"])

    lines = [
        "# NFR-08 -- context window compression evidence",
        "",
        f"Before: {len(messages)} messages, ~{tokens_before} tokens (approx.)",
        f"After: {len(result['summarized_messages'])} messages, ~{tokens_after} tokens (approx.)",
        "",
        "Running summary:",
        "",
        f"> {result['context']['running_summary'].summary}",
    ]
    (trace_dir / "context_before_after.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return {**result, "tokens_before": tokens_before, "tokens_after": tokens_after}


if __name__ == "__main__":
    asyncio.run(run_demo())
    print((_REPO_ROOT / "traces" / "context_before_after.md").relative_to(_REPO_ROOT).as_posix())
