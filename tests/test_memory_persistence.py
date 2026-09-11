"""AC-07: memory persists across sessions. Form 1 — in-process: session A writes,
all objects torn down, session B opens a FRESH store at the same path and recalls
by exact key (no LLM). Form 2 — two real `python` processes via the committed
script; its combined stdout is the committed traces/memory_persistence.log."""

import subprocess
import sys
from pathlib import Path

from pa_copilot.memory.store import memory_store

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "traces" / "memory_persistence.log"
NS = ("pa", "member", "M100001")
REC = {"content": "Determination: lumbar MRI 72148 APPROVED; cited PA-MRI-LUMBAR step-therapy.",
       "importance": "critical", "disposition": "approve"}


def test_in_process_teardown_then_fresh_store_recalls(tmp_path, fake_embedder):
    db = tmp_path / "persist.db"
    with memory_store(db, embedder=fake_embedder) as a:
        a.put(NS, "det-1", REC)
    # `a` and its connection are closed here.
    with memory_store(db, embedder=fake_embedder) as b:
        got = b.get(NS, "det-1")
    assert got is not None
    assert got.value["disposition"] == "approve"
    assert got.value["content"] == REC["content"]


def test_committed_log_matches_a_fresh_two_process_run(tmp_path):
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_persistence_test.py"),
         "--db", str(tmp_path / "x.db"), "--stdout-only"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout == LOG.read_text(encoding="utf-8")
