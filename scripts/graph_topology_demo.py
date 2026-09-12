"""AC-01 evidence: a byte-stable structural dump of the compiled graph's topology
(design.md §3.1). Builds `make_graph(...)` with `FakeToolCallingModel` and
in-memory store/checkpointer -- no real files needed, this is a structural
dump, not a run -- and writes the result to `traces/graph_topology.txt`.

Uses `graph.get_graph().draw_mermaid()`, not `.draw_ascii()` as design.md
originally sketched: verified empirically (this task) that `.draw_ascii()`'s
node layout differs on every call, including five straight in-process rebuilds
of the identical graph. Traced the cause to the installed `grandalf` (the
library `langchain_core.runnables.graph_ascii.draw_ascii()` delegates layout
to): `grandalf.graphs.Vertex` defines neither `__hash__` nor `__eq__`, so it
falls back to `object`'s default identity-based hash (confirmed by reading the
installed source: `Vertex.__hash__ is object.__hash__` and `Vertex.__eq__ is
object.__eq__` both hold). Grandalf's internal bookkeeping (e.g. its `Poset`
sets and the plain `set()`s built during layout) is keyed on these vertices, so
their iteration order follows each `Vertex` object's memory address rather
than any content property -- and `draw_ascii()` builds a fresh set of `Vertex`
objects, at new addresses, on every single call, which is exactly why even
in-process reruns of the same graph reordered. (An earlier, since-corrected
version of this note attributed the non-determinism to `PYTHONHASHSEED`
string-hash randomization; that can't be the real mechanism, since
`PYTHONHASHSEED` is fixed for the lifetime of a process and can't explain
same-process variation across the five rebuilds.) `.draw_mermaid()`'s output
(fixed node-declaration and edge order, no grandalf involved) was confirmed
byte-identical across the same in-process and cross-`PYTHONHASHSEED` checks,
so it is the byte-stable choice for a committed trace.

    python scripts/graph_topology_demo.py

``sys.path[0]`` is ``scripts/`` when invoked this way, so put the repo root
(and ``src``) on the path first (mirrors ``scripts/run_persistence_test.py`` /
``scripts/summarization_demo.py``).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _fake_model():
    """Bare-import the shared test double (mirrors `summarization_demo.py`'s
    `_fake_model()`): `tests/` is not a package, so this is
    `from _fakes import FakeToolCallingModel` with `tests/` put on `sys.path`
    at call time -- not `from tests._fakes import ...`. No node is actually
    invoked (this only introspects the compiled graph's static shape), so the
    fake never needs a scripted response."""
    sys.path.insert(0, str(_REPO_ROOT / "tests"))
    from _fakes import FakeToolCallingModel  # noqa: PLC0415

    return FakeToolCallingModel()


async def build_topology_dump() -> str:
    from langgraph.checkpoint.memory import InMemorySaver  # noqa: PLC0415
    from langgraph.store.memory import InMemoryStore  # noqa: PLC0415

    from pa_copilot.graph import make_graph  # noqa: PLC0415

    graph = await make_graph(
        store=InMemoryStore(),
        checkpointer=InMemorySaver(),
        mcp_tools=[],
        model=_fake_model(),
    )
    return graph.get_graph().draw_mermaid()


async def main() -> None:
    dump = await build_topology_dump()
    out = _REPO_ROOT / "traces" / "graph_topology.txt"
    out.write_text(dump, encoding="utf-8", newline="\n")
    print(out.relative_to(_REPO_ROOT).as_posix())


if __name__ == "__main__":
    asyncio.run(main())
